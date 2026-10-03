import unittest
from unittest.mock import patch
from pydantic import ValidationError
from geoimpact.core import Location, Event, Extraction, distance_km, scenario, circle, validate_evidence
from geoimpact.ingest import validate_url, prepare, MAX_CHARS

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.a = Location(name='Singapore',latitude=1.3521,longitude=103.8198)
    def test_coordinates(self):
        with self.assertRaises(ValidationError):
            Location(name='Invalid',latitude=91,longitude=0)
    def test_distance(self):
        self.assertEqual(distance_km(self.a,self.a),0)
        b=Location(name='Antipode',latitude=-self.a.latitude,longitude=self.a.longitude-180)
        self.assertAlmostEqual(distance_km(self.a,b),20015.114,places=2)
    def test_scenario(self):
        self.assertTrue(scenario(self.a,self.a,10,'supplier')['inside_assumed_radius'])
        self.assertIn('Unknown',scenario(self.a,self.a,10,'None / unknown')['indirect_exposure'])
        with self.assertRaises(ValueError): scenario(self.a,self.a,0,'supplier')
    def test_radius_geometry(self):
        a=Location(name='Dateline',latitude=0,longitude=179)
        lats,lons=circle(a,500)
        self.assertIn(None,lons)
        for lat,lon in zip(lats,lons):
            if lat is not None:
                self.assertAlmostEqual(distance_km(a,Location(name='Point',latitude=lat,longitude=lon)),500,places=5)
    def test_grounding(self):
        event=Event(title='Flood',event_type='natural_hazard',evidence_quote='Flooding closed the warehouse.',reported_scope='Unknown',possible_impacts=[])
        result=Extraction(events=[event])
        validate_evidence(result,'Flooding closed the warehouse.')
        with self.assertRaises(ValueError): validate_evidence(result,'A sunny day at the warehouse.')
        event.location_name='Singapore'
        with self.assertRaises(ValueError): validate_evidence(result,'Flooding closed the warehouse.')
    def test_private_url(self):
        for address in ['127.0.0.1','10.0.0.1','169.254.169.254','::1']:
            with patch('socket.getaddrinfo',return_value=[(2,1,6,'',(address,80))]):
                with self.assertRaises(ValueError): validate_url('http://example.com')
        with self.assertRaises(ValueError): validate_url('file:///etc/passwd')
    def test_public_url(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('8.8.8.8',443))]):
            validate_url('https://example.com/article')
    def test_document_limits(self):
        with self.assertRaises(ValueError): prepare('short','test')
        result=prepare('a'*(MAX_CHARS+10),'test')
        self.assertTrue(result['truncated']); self.assertEqual(len(result['text']),MAX_CHARS)
        self.assertEqual(result['id'],prepare('a'*(MAX_CHARS+10),'other')['id'])

if __name__ == '__main__': unittest.main()
