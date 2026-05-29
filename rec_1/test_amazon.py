import urllib.request
import gzip

url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'
print("Checking URL")
try:
    urllib.request.urlopen(url)
    print("URL is accessible")
except Exception as e:
    print(f"Error: {e}")
