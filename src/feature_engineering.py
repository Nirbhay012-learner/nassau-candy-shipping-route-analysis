import pandas as pd

FACTORY_MAPPING = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",

    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory"
}
FACTORY_COORDINATES = {
    "Lot's O' Nuts": {
        "Latitude": 32.881893,
        "Longitude": -111.768036
    },
    "Wicked Choccy's": {
        "Latitude": 32.076176,
        "Longitude": -81.088371
    },
    "Sugar Shack": {
        "Latitude": 48.11914,
        "Longitude": -96.18115
    },
    "Secret Factory": {
        "Latitude": 41.446333,
        "Longitude": -90.565487
    },
    "The Other Factory": {
        "Latitude": 35.1175,
        "Longitude": -89.971107
    }
}
def add_factory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Factory"] = df["Product Name"].map(FACTORY_MAPPING)

    return df
def add_factory_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Factory Latitude"] = df["Factory"].map(
        lambda x: FACTORY_COORDINATES[x]["Latitude"]
    )

    df["Factory Longitude"] = df["Factory"].map(
        lambda x: FACTORY_COORDINATES[x]["Longitude"]
    )

    return df
def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Route State"] = (
        df["Factory"] + " → " + df["State/Province"]
    )

    df["Route Region"] = (
        df["Factory"] + " → " + df["Region"]
    )

    return df
def add_high_cost_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cost_threshold = df["Cost"].quantile(0.75)

    df["High_Cost"] = (
        df["Cost"] >= cost_threshold
    ).astype(int)

    return df
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_factory(df)
    df = add_factory_coordinates(df)
    df = add_route_features(df)
    df = add_high_cost_target(df)

    return df