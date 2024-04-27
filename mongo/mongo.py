
# make sure to load secret
def new_mongo_client():
    from pymongo.mongo_client import MongoClient
    import os
    uri = "mongodb+srv://dotsangjason:"+ os.environ["MONGO_PASSWORD"] + "@srma.nhlebom.mongodb.net/?retryWrites=true&w=majority&appName=SRMA"
    # Create a new client and connect to the server
    mongo_client = MongoClient(uri)
    # Send a ping to confirm a successful connection
    mongo_client.admin.command('ping')
    
    return mongo_client