import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url


class Amazon(Dataset):

    __corefile__ = {
        "Books": "ratings_Books.csv",
        "Electronics": "ratings_Electronics.csv",
        "Movies": "ratings_Movies_and_TV.csv",
        "CDs": "ratings_CDs_and_Vinyl.csv",
        "Clothing": "ratings_Clothing_Shoes_and_Jewelry.csv",
        "Home": "ratings_Home_and_Kitchen.csv",
        "Kindle": "ratings_Kindle_Store.csv",
        "Sports": "ratings_Sports_and_Outdoors.csv",
        "Phones": "ratings_Cell_Phones_and_Accessories.csv",
        "Health": "ratings_Health_and_Personal_Care.csv",
        "Toys": "ratings_Toys_and_Games.csv",
        "VideoGames": "ratings_Video_Games.csv",
        "Tools": "ratings_Tools_and_Home_Improvement.csv",
        "Beauty": "ratings_Beauty.csv",
        "Apps": "ratings_Apps_for_Android.csv",
        "Office": "ratings_Office_Products.csv",
        "Pet": "ratings_Pet_Supplies.csv",
        "Automotive": "ratings_Automotive.csv",
        "Grocery": "ratings_Grocery_and_Gourmet_Food.csv",
        "Patio": "ratings_Patio_Lawn_and_Garden.csv",
        "Baby": "ratings_Baby.csv",
        "Music": "ratings_Digital_Music.csv",
        "MusicalInstruments": "ratings_Musical_Instruments.csv",
        "InstantVideo": "ratings_Amazon_Instant_Video.csv",
    }

    url_prefix = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"

    def download(self, category):
        download_url(
            self.url_prefix + self.__corefile__[category],
            self.rootdir.joinpath(self.__corefile__[category]),
        )

    def transform(self, category, rating_threshold):
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__[category]),
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        df = df[df["rating"] >= rating_threshold].drop("rating", axis=1)
        return df
