# type: ignore
# flake8: noqa
# replacing ? symbol in specific columns with other values
insurance['collision_type'] = insurance['collision_type'].replace('?', 'No Collision')
insurance['property_damage'] = insurance['property_damage'].replace('?', 'Unsure')
insurance['police_report_available'] = insurance['police_report_available'].replace('?', 'In Progress')


