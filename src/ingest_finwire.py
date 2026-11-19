import requests

nbp_url = "https://public-api.finwire.pl/v1/series/interest-rate/stopa-referencyjna"
respose = requests.get(url=nbp_url, params={ "days" : 4000 }, timeout=30)

print(respose.status_code)

data = respose.json()
print(data)
