import discord
import aiohttp

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

async def fetch(session, url, json):
    async with session.post(url, json=json) as response:
        return await response.json()

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    async with aiohttp.ClientSession() as session:
        response_json = await fetch(session, 'http://localhost:5005/webhooks/rest/webhook', {"sender": str(message.author.id), "message": message.content})
        try:
            response_text = response_json[0]['text']
            await message.channel.send(response_text)
        except IndexError:
            await message.channel.send("I'm not sure how to respond to that.")

client.run('MTIyNjk1MDUyNzA1MzU5NDYyNA.GN2Iqd.gq1-O8hrvvQ3pa402GQu07A0Th5Z_5YpmCeRnc')  
