import pylsl
from typing import List, Dict, Union

#===========================================#
# ===========================================#
#       Writing to stream info              #
# ===========================================#

def add_channel_names(stream_info, channel_names):
    """Adds channel metadata to LSL stream header."""
    add_channel_metadata(stream_info, channel_names)


def add_channel_metadata(stream_info, channel_labels, channel_units=None, channel_impedances=None, channel_types=None):
    """Adds channel metadata to LSL stream header."""
    desc = stream_info.desc()
    channels = desc.append_child('channels')

    for i, label in enumerate(channel_labels):
        channel = channels.append_child('channel')
        channel.append_child_value('label', label)
        if channel_units is not None:
            channel.append_child_value('unit', channel_units[i])
        if channel_impedances is not None:
            channel.append_child_value('impedance', channel_impedances[i])
        if channel_types is not None:
            channel.append_child_value('type', channel_types[i])


def add_mappings(stream_info, names, enums):
    """Adds mappings to LSL stream header. Adds new child 'mappings' to 
        stream info desc with a child per name-enum pair. The mappings 
        themselves are stored as 'keys' and 'values' lists in each child."""
    desc = stream_info.desc()
    mappings = desc.append_child('mappings')

    for name, enum in zip(names, enums):
        mapping = mappings.append_child(name)
        for item in enum:
            mapping.append_child_value(item.name, str(item.value))


def add_parameters(stream_info, parameters: dict):
    """ Adds module Parameters to LSL stream header.
    """

    desc = stream_info.desc()
    parameters_xml = desc.append_child('parameters')

    for name, parameter in parameters.items():
        parameters_xml.append_child_value(name, str(parameter.getValue()))


# ===========================================#
#       Finding in stream info              #
# ===========================================#
def find_stream(xdf_data, stream_names) -> Dict[str, dict]:
    """Return dictionary of format stream_name: (index, stream)
        if stream_name is found in xdf data."""
    try:
        streams = {stream['info']['name'][0]: stream
                   for (ix, stream) in enumerate(xdf_data)
                   if stream['info']['name'][0] in stream_names}

        if len(stream_names) > len(streams):
            raise KeyError("""Error in {}. Could not find {} in streams."""
                           .format(find_stream.__name__, *(set(stream_names) - set(streams.keys()))))

        return streams

    except TypeError:
        raise TypeError("""Error in {}. 
            Could not find streams in xdf_data[ix]['info']['name']"""
                        .format(find_stream.__name__))


def find_channel_index(stream, channel_names):
    """Find channels by name and return indices for accessing slices of
        time series data."""
    try:
        channel_list = stream['info']['desc'][0]['channels'][0]['channel']
        channels = [channel_list[c]['label'][0] for c in range(len(channel_list))]

        # throws a ValueError if requested channel name is not in stream channels
        indices = list(map(channels.index, channel_names))
        return indices

    except TypeError:
        raise TypeError("""Error in {}. 
            Could not find channels in stream['info']['desc'][0]['channels'][0]['channel']"""
                        .format(find_channel_index.__name__))


def get_channel_labels(stream_info: pylsl.StreamInfo) -> List[str]:
    """Extract channel labels from pylsl.StreamInfo object."""

    channel_item = stream_info.desc().child("channels").first_child()

    # special treatment for SMARTING (Mbraintrain) LSL stream: Has a "type" XML-Tag in
    # the "channels" section that does represent a channel
    if not isinstance(channel_item.name(), str) or channel_item.name().lower() != 'channel':
        channel_item = channel_item.next_sibling()

    channel_labels = [channel_item.child_value("label")]
    while not channel_item.next_sibling().empty():
        channel_item = channel_item.next_sibling()
        if not isinstance(channel_item.name(), str) or channel_item.name().lower() != 'channel':
            continue
        channel_labels.append(channel_item.child_value("label"))
    return channel_labels


def get_parameters_from_xdf_stream(xdf_stream) -> Dict[str, Union[str, int, float, bool]]:
    """
        Extract parameters as a dict from metadata of an XDF data object.
    """

    parameters = {}

    if 'parameters' not in xdf_stream['info']['desc'][0].keys():
        return {}

    keys = list(xdf_stream['info']['desc'][0]['parameters'][0].keys())

    for k in keys:
        parameters[k] = xdf_stream['info']['desc'][0]['parameters'][0][k][0]

    return parameters
