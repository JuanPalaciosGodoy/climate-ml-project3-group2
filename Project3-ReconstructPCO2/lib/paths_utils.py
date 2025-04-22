class SavingPaths(object):
    def __init__(self, your_username:str, owner_username:str, init_date:str, fin_date:str, grid_search_approach:str='nmse'):
        self.your_username = your_username
        self.owner_username = owner_username
        self.grid_search_approach = grid_search_approach
        self.init_date = init_date
        self.fin_date = fin_date

    @property
    def inputs_path(self) -> str:
        """
        input data path for ML
        """
        return f"gs://leap-persistent/{self.your_username}/pco2_residual/post01_xgb_inputs_parquet"

    @property
    def _zarr_dir(self) -> str:
        """
        zarr directory
        """
        return "gs://leap-persistent/abbysh/zarr_files_"
    
    @property
    def socat_path(self) -> str:
        """
        socat data file
        """
        return f"{self._zarr_dir}/socat_mask_feb1982-dec2023.zarr"

    @property
    def xco2_path(self) -> str:
        """
        atmospheric xco2 file
        """
        return f"{self._zarr_dir}/xco2_cmip6_183501-224912_monthstart.zarr"

    @property
    def topo_path(self) -> str:
        """
        topo mask
        """
        return f"{self._zarr_dir}/GEBCO_2014_1x1_global.zarr"

    @property
    def lsmask_path(self) -> str:
        """
        land-sea mask
        """
        return f"{self._zarr_dir}/lsmask.zarr"

    @property
    def path_seeds(self) -> str:
        """
        random seeds for ML
        """
        return "gs://leap-persistent/abbysh/pickles/random_seeds.npy"

    @property
    def ensemble_dir(self) -> str:
        """
        directory of regridded members from notebook 00
        """
        return "gs://leap-persistent/abbysh/pco2_all_members_1982-2023/00_regridded_members"

    @property
    def output_dir(self) -> str:
        """
        where to save machine learning results
        """
        return f'gs://leap-persistent/{self.your_username}/{self.owner_username}/pco2_residual/{self.grid_search_approach}/post02_xgb'

    @property
    def model_output_dir(self) -> str:
        """
        where to save ML models
        """
        return f"{self.output_dir}/trained"

    @property
    def recon_output_dir(self) -> str:
        """
        where to save ML reconstructions
        """
        return f"{self.output_dir}/reconstructions"

    @property
    def metrics_output_dir(self) -> str:
        """
        where to save performance metrics
        """
        return f"{self.output_dir}/metrics"

    @property
    def test_perform_fname(self) -> str:
        """
        path for test performance metrics
        """
        return f"{self.metrics_output_dir}/xgb_test_performance_{self.init_date}-{self.fin_date}.csv"

    @property
    def unseen_perform_fname(self) -> str:
        """
        path for unseen performance metrics
        """
        return f"{self.metrics_output_dir}/xgb_unseen_performance_{self.init_date}-{self.fin_date}.csv"

    @property
    def model_save_dir(self) -> str:
        """
        where to save .json model file
        """
        return f"{self.output_dir}/saved_models_{self.init_date}-{self.fin_date}"

    @property
    def model_local_save_dir(self) -> str:
        """
        local directory to save models
        """
        return "output/model_saved"

    def load_model_local_path(self, ens:str, member:str, extension:str) -> str:
        """
        local directory to load model
        """

        return f"{self.model_local_save_dir}/model_pCO2_2D_{ens}_{member.split('_')[-1]}_mon_1x1_{self.init_date}_{self.fin_date}.{extension}"

    def inputs_ens_member_path(self, ens:str, member:str) -> str:
        """
        input ens member directory
        """
        data_dir = f"{self.inputs_path}/{ens}/{member}"
        fname = f"MLinput_{ens}_{member.split('_')[-1]}_mon_1x1_{self.init_date}_{self.fin_date}.parquet"
        return f"{data_dir}/{fname}"

















        
        