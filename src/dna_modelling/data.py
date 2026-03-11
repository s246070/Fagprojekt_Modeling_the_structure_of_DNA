from pathlib import Path
from urllib.request import urlretrieve
import scanpy as sc

# This class is responsible for downloading and loading the data. 
# It ensures that the data is downloaded to a specific directory 
# and provides a method to load the data into either memory or hard drive.
class Data():

    # The constructor initializes the data directory, the path to the data file, 
    # and the backup URL for downloading the data if it is not already present.
    def __init__(self):
        self.data_directory = Path("data")
        self.data_path = self.data_directory / "lung_atlas_preprocessed.h5ad"
        self.backup_url = "https://figshare.com/ndownloader/files/52859288"

    # This method checks if the data file exists at the specified path. 
    # If it does not exist, it creates the necessary directories and downloads the data from the backup URL.
    def ensure_data_downloaded(self):
        self.data_directory.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            print(f"Downloading datset to {self.data_path}...")
            urlretrieve(self.backup_url, self.data_path)

    # This method loads the data from the specified path. 
    # If the 'backed' parameter is set to True, it loads the data into the hard drive.
    # If 'backed' is False, it loads the entire dataset into memory.
    def load_data(self, backed=True):
        self.ensure_data_downloaded()

        if backed:
            return sc.read_h5ad(self.data_path, backed="r")
        return sc.read_h5ad(self.data_path)