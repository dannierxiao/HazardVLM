HAZARD_TYPE_DICT = { # x8 hazard types
                'No hazardous event': 0,
                'Hazardous turn manoeuvre': 1,
                'Hazardous start stop': 2,
                'Hazardous lane change': 3,
                'Hazardous lane exit': 4,
                'Hazardous oncoming approach': 5,
                'Hazardous pedestrian on': 6,
                '<unknown>': 7,
                }


LOCATION_TYPE_DICT = { # x5 relation types
                        'no hazard': 0,
                        'straight ahead': 1,
                        'on right': 2,
                        'on left': 3,
                        '<unknown>': 4,
                    }


ACTOR_TYPE_DICT = { # x9 actor types
                'no hazard': 0,
                'car': 1,
                'truck': 2, 
                'motorcycle': 3,
                'actors': 4,
                'pedestrian': 5,
                'bus': 6,
                'bicycle': 7,
                '<unknown>': 8,
                }