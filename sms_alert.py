import os
from twilio.rest import Client

account_sid = "ACd27f425faba63d48eda6f9f5455d235d"
auth_token = "8e208dc8869025e0ced14fbaac8cc99d"


client = Client(account_sid,auth_token)

client.messages.create(
    to="+64220950208",
    from_="+61447258865",
    body="Testing123"
)
