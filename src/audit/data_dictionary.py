"""Documented meanings and domains of the source columns."""

COLUMN_METADATA = {
    "track_id": {
        "Description": "The unique Spotify ID for the track.",
        "Type": "Identifier",
        "Unit or scale": "Unitless",
        "Documented domain": "Alphanumeric string",
    },
    "artists": {
        "Description": "The artists who performed the track. Multiple artists are separated by a semicolon (;).",
        "Type": "Text with semicolon-separated values",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "album_name": {
        "Description": "The album name in which the track appears.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "track_name": {
        "Description": "The name of the track.",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Text",
    },
    "popularity": {
        "Description": "The popularity of a track. Calculated by algorithm and based, in the most part, on the total number of plays the track has had and how recent those plays are.",
        "Type": "Numeric, recorded as integers",
        "Unit or scale": "Popularity score",
        "Documented domain": "Between 0 and 100 (100 being the most popular)",
    },
    "duration_ms": {
        "Description": "Duration of the song.",
        "Type": "Numeric, recorded as integers",
        "Unit or scale": "Milliseconds",
        "Documented domain": "Limits not specified",
    },
    "explicit": {
        "Description": "Whether or not the track has explicit lyrics.",
        "Type": "Boolean",
        "Unit or scale": "Unitless",
        "Documented domain": "True or False (False indicates no explicit content, or that the explicit status is unknown)",
    },
    "danceability": {
        "Description": "Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "energy": {
        "Description": "Perceived intensity and activity of the song. Energetic tracks feel fast, loud, and noisy.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "key": {
        "Description": "The key the track is in. Integers map to pitches using standard Pitch Class notation (e.g., 0 = C, 1 = C♯/D♭, 2 = D).",
        "Type": "Categorical integer",
        "Unit or scale": "Pitch class",
        "Documented domain": "Between -1 and 11 (-1 if no key was detected)",
    },
    "loudness": {
        "Description": "The overall loudness of a track in decibels (dB), averaged across the entire track. Useful for comparing relative loudness of tracks.",
        "Type": "Continuous numeric",
        "Unit or scale": "Decibels (dB)",
        "Documented domain": "Typically between -60 and 0",
    },
    "mode": {
        "Description": "Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.",
        "Type": "Binary integer",
        "Unit or scale": "Unitless",
        "Documented domain": "1 for Major, 0 for Minor",
    },
    "speechiness": {
        "Description": "Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book), the closer to 1.0.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (Values > 0.66 probably represent entirely spoken words)",
    },
    "acousticness": {
        "Description": "A confidence measure of whether the track is acoustic.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (1.0 represents high confidence)",
    },
    "instrumentalness": {
        "Description": "Predicts whether a track contains no vocals. 'Ooh' and 'aah' sounds are treated as instrumental. Rap or spoken word are vocal.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0 (Values closer to 1.0 indicate no vocal content)",
    },
    "liveness": {
        "Description": "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "valence": {
        "Description": "A measure describing the musical positiveness conveyed by a track. High valence sounds more positive (happy, cheerful), low valence sounds negative (sad, depressed).",
        "Type": "Continuous numeric",
        "Unit or scale": "Unitless",
        "Documented domain": "Between 0.0 and 1.0",
    },
    "tempo": {
        "Description": "The overall estimated tempo of a track in beats per minute (BPM).",
        "Type": "Continuous numeric",
        "Unit or scale": "Beats per minute (BPM)",
        "Documented domain": "Limits not specified",
    },
    "time_signature": {
        "Description": "An estimated time signature. It specifies how many beats are in each bar (or measure).",
        "Type": "Categorical integer",
        "Unit or scale": "Beats per bar",
        "Documented domain": "Between 3 and 7 (e.g., 3/4 to 7/4 time signature)",
    },
    "track_genre": {
        "Description": "Text labels for musical genres",
        "Type": "Categorical string",
        "Unit or scale": "Unitless",
        "Documented domain": "Text labels; observed categories are reported in Genre Representation",
    }
}
