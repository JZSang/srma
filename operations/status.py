
from modal_references import status_tracker_global_dictionary

def get_status_impl(mode):
    return status_tracker_global_dictionary[mode].__dict__