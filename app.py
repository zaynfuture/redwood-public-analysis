"""Redwood Public Analysis — Streamlit entry point."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import plotly.graph_objects as go
import streamlit as st
from geoimpact.core import Location, Event, circle, scenario
from geoimpact.ingest import fetch, pdf_text, prepare
from geoimpact.extract import extract, make_pipeline, classify_batch

ROOT = Path(__file__).resolve().parent

@st.cache_resource(max_entries=2)
def cached_pipeline(task, model_id, local_only, version=0):
    return make_pipeline(task, model_id, local_only)

@st.cache_data(max_entries=64, show_spinner=False)
def cached_classification(text, model_path, version):
    classifier = cached_pipeline('text-classification', model_path, True, version)
    return classify_batch([text], classifier)[0]

CITIES = {
    'Singapore': (1.3521, 103.8198), 'Bangkok': (13.7563, 100.5018), 'Shanghai': (31.2304, 121.4737),
    'London': (51.5074, -0.1278), 'New York': (40.7128, -74.0060),
    'Tokyo': (35.6762, 139.6503), 'Rotterdam': (51.9244, 4.4777),
    'Sydney': (-33.8688, 151.2093), 'Cape Town': (-33.9249, 18.4241)
}
DEMO_TEXT = ('Fictional training scenario. Flooding in Singapore closed a coastal warehouse '
             'and delayed truck deliveries on Monday. The operator reported that two loading '
             'bays were unavailable. No damage outside the warehouse was confirmed.')
DEMO_EVENT = Event(title='Flooding interrupts warehouse operations', event_type='natural_hazard',
    location_name='Singapore', location_quote='Flooding in Singapore',
    evidence_quote='Flooding in Singapore closed a coastal warehouse and delayed truck deliveries on Monday.',
    reported_scope='One coastal warehouse; two loading bays unavailable.',
    possible_impacts=['If your deliveries use this warehouse, dispatch may be delayed.',
                      'If you have alternate storage, rerouting could reduce disruption.'])

st.set_page_config(page_title='Redwood Public Analysis', page_icon='🌐', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220; color: #e9eef5;}
[data-testid="stSidebar"] {background: #111d2e;}
h1 {letter-spacing: -1.5px;} .stCaption {color: #a1b6c9;}
</style>''', unsafe_allow_html=True)
st.caption('REDWOOD  /  PUBLIC ANALYSIS')
st.title('Understand the event. Explore your exposure.')
st.write('Turn a news report into source-grounded events and location-specific business scenarios.')

with st.expander('How we fine-tune and evaluate the model'):
    st.markdown((ROOT / 'MODEL_WORKFLOW.md').read_text())
    st.download_button('Download training and evaluation guide',
        (ROOT / 'MODEL_WORKFLOW.md').read_text(), 'MODEL_WORKFLOW.md', 'text/markdown')
    evaluation_report = ROOT / 'artifacts' / 'evaluation-test.json'
    if evaluation_report.exists():
        evaluation = json.loads(evaluation_report.read_text())
        st.caption('Measured evaluation of the saved checkpoint' + (' (synthetic smoke dataset)' if evaluation.get('synthetic') else ''))
        st.json(evaluation)

with st.sidebar:
    st.header('Analysis workspace')
    mode = st.radio('Source mode', ['Demo scenario', 'News URL', 'Upload PDF', 'Paste text'])
    model_name = st.selectbox('Hugging Face extraction model', ['Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-3B-Instruct'])
    local_only = st.checkbox('Use downloaded models only', value=False, key='model_offline_v2',
        help='Leave off on Streamlit Cloud. Its model cache is separate from your laptop and may be empty after a restart.')
    st.caption('First analysis downloads the selected public model if needed. Later analyses reuse the cached model. Offline mode requires an existing server-side cache.')
    use_classifier = st.checkbox('Show fine-tuned classifier second opinion')
    st.divider()
    st.caption('Academic prototype · Evidence first · English interface')
    st.caption('News and PDF contents are source material, never application instructions.')

left, right = st.columns([1.15, 1])
with left:
    st.subheader('01 / Add a source')
    url, upload, pasted = '', None, ''
    if mode == 'Demo scenario':
        st.info('Fictional example. No live news or LLM inference is used in demo mode.')
        st.write(DEMO_TEXT)
    elif mode == 'News URL':
        url = st.text_input('Public article or PDF URL', placeholder='https://news.example.com/article')
    elif mode == 'Upload PDF':
        upload = st.file_uploader('Text-based news PDF (8 MB / 40 pages maximum)', type=['pdf'])
    else:
        pasted = st.text_area('News text', height=160)
    if st.button('Analyze source', type='primary'):
        # Clear stale results before a potentially failing request.
        st.session_state.pop('analysis', None)
        try:
            with st.spinner('Reading and extracting supported events…'):
                if mode == 'Demo scenario':
                    source = prepare(DEMO_TEXT, 'Original fictional demonstration')
                    events = [DEMO_EVENT]
                else:
                    if mode == 'News URL':
                        text, origin = fetch(url)
                    elif mode == 'Upload PDF':
                        if upload is None:
                            raise ValueError('Upload a PDF first.')
                        text, origin = pdf_text(upload.getvalue()), upload.name
                    else:
                        text, origin = pasted, 'User-provided text'
                    source = prepare(text, origin)
                    with st.spinner('Loading model — the first run may download Hugging Face weights…'):
                        generator = cached_pipeline('text-generation', model_name, local_only)
                    events = extract(source['text'], generator).events
                st.session_state.analysis = {'source': source, 'events': [e.model_dump() for e in events],
                    'mode': mode, 'extractor': 'fixture' if mode == 'Demo scenario' else model_name,
                    'created_at': datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            st.error(f'Analysis could not finish: {exc}')

with right:
    st.subheader('02 / Define your location')
    city = st.selectbox('Business location', list(CITIES) + ['Custom coordinates'])
    if city == 'Custom coordinates':
        user_name = st.text_input('Location name', 'My business')
        user_lat = st.number_input('Latitude', -90., 90., 0., key='user_lat')
        user_lon = st.number_input('Longitude', -180., 180., 0., key='user_lon')
    else:
        user_name, (user_lat, user_lon) = city, CITIES[city]
    dependency = st.selectbox('Relevant business dependency',
        ['None / unknown', 'supplier', 'shipping route', 'customer market', 'office / staff'])

analysis = st.session_state.get('analysis')
if not analysis:
    st.divider()
    st.info('Start with the fictional demo, or add a news source to build your analysis.')
    st.stop()
if analysis['mode'] != mode:
    st.warning('The results below belong to the previous source mode. Analyze the new source to replace them.')
if not analysis['events']:
    st.warning('No source-supported events were extracted. Try another report or a shorter excerpt.')
    st.stop()
source = analysis['source']
if source['truncated']:
    st.warning('This document exceeded the input limit. Only the first 24,000 characters were analyzed.')
st.divider()
idx = st.selectbox('Extracted event', range(len(analysis['events'])),
                   format_func=lambda i: analysis['events'][i]['title'])
event = Event.model_validate(analysis['events'][idx])
st.subheader(event.title)
a, b = st.columns(2)
with a:
    st.caption('REPORTED IN THE SOURCE')
    st.write(f'Event type: **{event.event_type.replace("_", " ")}**')
    st.write(event.reported_scope)
    st.text(event.evidence_quote)
    st.caption(f"Evidence: {source['id']} · Location: {event.location_name or 'Unknown'}")
with b:
    st.caption('POSSIBLE BUSINESS EFFECTS · LLM HYPOTHESES')
    for impact in event.possible_impacts:
        st.write('• ' + impact)
    st.caption('These hypotheses are not verified consequences or probability estimates.')
if use_classifier:
    path = ROOT / 'artifacts' / 'transformer'
    if not (path / 'config.json').exists():
        st.warning('No fine-tuned checkpoint found. Run the training command in the README first.')
    else:
        try:
            prediction = cached_classification(event.evidence_quote, str(path), (path / 'config.json').stat().st_mtime_ns)
            st.write('Fine-tuned classifier:', prediction)
            report_path = ROOT / 'artifacts' / 'transformer-metrics.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            st.caption(('Synthetic smoke-trained model. ' if report.get('synthetic', True) else '') + 'Scores are uncalibrated; this is not independent verification.')
        except Exception as exc:
            st.error(f'Classifier unavailable: {exc}')

st.subheader('03 / Explore a geographic scenario')
st.caption('Confirm the event location. Coordinates are user inputs, not verified LLM outputs.')
known = event.location_name if event.location_name in CITIES else 'Custom coordinates'
key = f"{source['id']}-{idx}"
event_city = st.selectbox('Event location coordinates', list(CITIES)+['Custom coordinates'],
    index=(list(CITIES)+['Custom coordinates']).index(known), key='city-'+key)
if event_city == 'Custom coordinates':
    event_name = st.text_input('Event place name', event.location_name or 'Unresolved location', key='name-'+key)
    c1, c2 = st.columns(2)
    elat = c1.number_input('Event latitude', -90., 90., 0., key='lat-'+key)
    elon = c2.number_input('Event longitude', -180., 180., 0., key='lon-'+key)
else:
    event_name, (elat, elon) = event_city, CITIES[event_city]
confirmed = st.checkbox('I have checked that these coordinates represent the reported event location.', key=f'confirm-{key}-{event_city}-{elat}-{elon}')
if not confirmed:
    st.info('Confirm event coordinates to display location-specific analysis.')
    st.stop()
event_location = Location(name=event_name, latitude=elat, longitude=elon)
user_location = Location(name=user_name or 'My business', latitude=user_lat, longitude=user_lon)
radius = st.slider('Assumed direct-impact radius (km)', 10, 3000, 100, 10)
st.caption('Move the slider to update the map. This is sensitivity analysis, not a physical spread forecast.')
result = scenario(event_location, user_location, radius, dependency)
fig = go.Figure()
lats, lons = circle(event_location, radius)
fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode='lines', name='Assumed footprint',
                           line=dict(color='#f5b65d', width=2)))
fig.add_trace(go.Scattergeo(lat=[elat, user_lat], lon=[elon, user_lon], mode='lines',
                           line=dict(color='#7f96af', width=1, dash='dot'), name='Location link'))
fig.add_trace(go.Scattergeo(lat=[elat], lon=[elon], text=[event_name], mode='markers+text',
                           marker=dict(size=13, color='#f5b65d'), name='Event'))
fig.add_trace(go.Scattergeo(lat=[user_lat], lon=[user_lon], text=[user_name], mode='markers+text',
                           marker=dict(size=9, color='#5cd5c4'), name='Your business', textposition='bottom center'))
fig.update_geos(projection_type='natural earth', projection_scale=1, fitbounds=False,
    center=dict(lat=0, lon=0), lataxis_range=[-90,90], lonaxis_range=[-180,180], showland=True, landcolor='#23364a',
    showocean=True, oceancolor='#101c2b', showcountries=True, countrycolor='#40556c',
    coastlinecolor='#40556c', bgcolor='#0b1220')
fig.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0b1220',
    font_color='#e9eef5', legend=dict(orientation='h'), uirevision='world')
st.plotly_chart(fig, width='stretch', theme=None)
c1, c2 = st.columns(2)
c1.metric('Distance from event', f"{result['distance_km']:,.1f} km")
c2.metric('Assumed footprint', 'Inside' if result['inside_assumed_radius'] else 'Outside')
st.write('**Direct exposure:** ' + result['direct_exposure'])
st.write('**Indirect exposure:** ' + result['indirect_exposure'])
st.caption(result['basis'])
export = {**analysis, 'selected_event': idx, 'event_location': event_location.model_dump(),
    'business_location': user_location.model_dump(), 'assumed_radius_km': radius,
    'declared_dependency': dependency, 'scenario': result,
    'exported_at': datetime.now(timezone.utc).isoformat()}
st.download_button('Download evidence and scenario (JSON)', json.dumps(export, indent=2),
                   'redwood-analysis.json', 'application/json')
with st.expander('Source provenance and analyzed text'):
    st.json({k:v for k,v in source.items() if k != 'text'})
    st.text(source['text'])
