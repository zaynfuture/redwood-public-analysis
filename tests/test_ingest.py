import unittest
from unittest.mock import patch, MagicMock
from geoimpact.ingest import fetch, pdf_text, MAX_BYTES

class IngestTests(unittest.TestCase):
    def response(self,body,kind='text/html',redirect=False):
        response=MagicMock()
        response.__enter__.return_value=response
        response.is_redirect=redirect
        response.headers={'Content-Type':kind,'Location':'http://127.0.0.1/private'}
        response.iter_content.return_value=[body]
        return response
    def test_html_cleanup(self):
        r=self.response(b'<html><nav>Ignore me</nav><article>Flooding closed the coastal warehouse and interrupted deliveries.</article><script>bad()</script></html>')
        with patch('geoimpact.ingest.validate_url'),patch('requests.Session.get',return_value=r):
            text,url=fetch('https://example.com/news')
        self.assertNotIn('Ignore',text);self.assertNotIn('bad()',text)
        self.assertIn('Flooding',text)
    def test_article_controls_excluded(self):
        r=self.response(b'<article><h1>Flood emergency</h1><button>Share Save</button><p>Flooding closed a warehouse and interrupted deliveries.</p><figure>Getty Images</figure></article>')
        with patch('geoimpact.ingest.validate_url'),patch('requests.Session.get',return_value=r):
            text,_=fetch('https://example.com/news')
        self.assertIn('Flood emergency',text)
        self.assertNotIn('Share Save',text)
        self.assertNotIn('Getty Images',text)
    def test_redirect_validated(self):
        with patch('geoimpact.ingest.validate_url',side_effect=[None,ValueError('Private')]) as validate,patch('requests.Session.get',return_value=self.response(b'',redirect=True)):
            with self.assertRaises(ValueError):fetch('https://example.com/news')
            self.assertEqual(validate.call_count,2)
    def test_download_limit(self):
        with patch('geoimpact.ingest.validate_url'),patch('requests.Session.get',return_value=self.response(b'x'*(MAX_BYTES+1))):
            with self.assertRaises(ValueError):fetch('https://example.com/news')
    def test_pdf_text_contract(self):
        reader=MagicMock();reader.is_encrypted=False
        page=MagicMock();page.extract_text.return_value='Flooding closed a warehouse and interrupted local truck deliveries.'
        reader.pages=[page]
        with patch('pypdf.PdfReader',return_value=reader):
            self.assertIn('Flooding',pdf_text(b'%PDF fixture'))
        page.extract_text.return_value=''
        with patch('pypdf.PdfReader',return_value=reader):
            with self.assertRaises(ValueError):pdf_text(b'%PDF fixture')

if __name__=='__main__':unittest.main()
