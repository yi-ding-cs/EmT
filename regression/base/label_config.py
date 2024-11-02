number_to_emotion_tag_dict = {
    "0": "Neutral",
    "1": "Anger",
    "2": "Disgust",
    "3": "Fear",
    "4": "Joy",
    "5": "Sadness",
    "6": "Surprise",
    "11": "Amusement",
    "12": "Anxiety"
}

emotion_tag_to_valence_class = {
    "Neutral": "Neutral valence",
    "Anger": "Unpleasant",
    "Disgust": "Unpleasant",
    "Fear": "Unpleasant",
    "Joy": "Pleasant",
    "Sadness": "Unpleasant",
    "Surprise": "Neutral valence",
    "Amusement": "Pleasant",
    "Anxiety": "Unpleasant"
}

emotion_tag_to_arousal_class = {
    "Neutral": "Calm",
    "Anger": "Activated",
    "Disgust": "Calm",
    "Fear": "Activated",
    "Joy": "Medium arousal",
    "Sadness": "Calm",
    "Surprise": "Activated",
    "Amusement": "Medium arousal",
    "Anxiety": "Activated"
}

valence_class_to_number = {
    "Unpleasant": 0,
    "Neutral valence": 1,
    "Pleasant": 2
}

arousal_class_to_number = {
    "Calm": 0,
    "Medium arousal": 1,
    "Activated": 2
}