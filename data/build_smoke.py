"""Original fictional examples for pipeline verification, not benchmark data."""
import csv
from pathlib import Path

examples = {
'natural_hazard': [
'Floodwater entered a warehouse and trucks could not reach its loading docks.',
'An earthquake damaged a regional road and deliveries were suspended.',
'A hurricane forced coastal shops to close before the storm arrived.',
'Wildfire smoke caused the distribution center to suspend outdoor loading.',
'Heavy rainfall triggered a landslide that blocked access to a factory.',
'A severe drought reduced the water available to food processors.',
'A volcanic eruption covered the airport runway with ash and flights stopped.',
'A blizzard left mountain roads impassable and isolated local retailers.'],
'conflict': [
'Armed fighting near a border checkpoint stopped commercial freight movements.',
'Military shelling damaged the industrial district during an armed conflict.',
'An armed group attacked cargo vehicles on the highway.',
'Clashes between soldiers forced the town market to close.',
'A naval confrontation caused commercial vessels to avoid the strait.',
'Hostilities near the port disrupted the arrival of merchant ships.',
'A missile strike during the war damaged an oil storage terminal.',
'Renewed combat led the operator to evacuate workers from its plant.'],
'public_health': [
'A disease outbreak prompted a temporary closure of a food processing facility.',
'Health officials confirmed infections among staff at a distribution center.',
'A contagious virus caused employee absences across local businesses.',
'An epidemic led authorities to restrict large public gatherings.',
'Public health inspectors linked a bacterial outbreak to contaminated water.',
'A flu outbreak reduced staffing capacity at the regional hospital.',
'Health authorities reported a measles cluster and issued isolation guidance.',
'A cholera outbreak led nearby restaurants to suspend water-dependent services.'],
'infrastructure': [
'A power outage stopped production at several factories in the city.',
'A burst water main closed an industrial road for emergency repairs.',
'A collapsed bridge interrupted the main freight route into town.',
'A telecommunications network failure prevented stores from processing payments.',
'A signaling malfunction shut down the commuter railway for several hours.',
'A damaged submarine cable disrupted internet connectivity for local companies.',
'A transformer failure left the commercial district without electricity.',
'A broken pipeline forced the utility company to halt gas service.'],
'trade_policy': [
'The government introduced an import tariff on manufactured steel products.',
'New export controls require licenses for shipments of advanced components.',
'Authorities announced sanctions restricting commerce with several companies.',
'A customs regulation increased documentation requirements for imported food.',
'The ministry imposed a quota on agricultural imports for the coming year.',
'A trade agreement removed duties on qualifying industrial machinery.',
'A new embargo prohibited specified technology exports to the destination.',
'The treasury announced additional customs duties on imported vehicle parts.'],
'other': [
'A software company unveiled a new scheduling application at its annual conference.',
'A local museum opened an exhibition of paintings from private collections.',
'A retailer announced a refreshed logo and a redesigned shopping website.',
'A university hosted a lecture on the history of regional architecture.',
'A publishing house released a novel by a first-time author.',
'A sports club announced its schedule of friendly matches for next summer.',
'A technology firm demonstrated a new keyboard at a product showcase.',
'A neighborhood library expanded its collection of historical photographs.']}

path = Path(__file__).with_name('smoke.csv')
with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['text','label','event_id','split','source_url','license','synthetic'])
    writer.writeheader()
    for label, texts in examples.items():
        for i, text in enumerate(texts):
            writer.writerow(dict(text=text,label=label,event_id=f'fiction-{label}-{i}',
                split='train' if i<6 else ('validation' if i==6 else 'test'),
                source_url='original-fiction',license='CC0-1.0',synthetic=True))
print(path)
