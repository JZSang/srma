def generate_unique_id():
    import uuid

    unique_id = str(uuid.uuid4())[:8]
    return unique_id
