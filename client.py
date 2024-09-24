import requests

# Define the URL
url = 'http://localhost:7070'

# Send a GET request
response = requests.get(url)

# Print the status code and response headers
print('Status Code:', response.status_code)
print('Response Headers:', response.headers)
print('Response Text:', response.text)
