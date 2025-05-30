## Data loader class for BOLD + SC matrices

import os, random
import numpy as np
import torch
import scipy.io
from whobpyt.utils.fc_tools import bold_to_fc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using DEVICE = {DEVICE}")

class BOLDDataLoader:
    def __init__(self, fmri_filename: str, sc_path: str, distance_matrix_path: str, chunk_length: int = 50):
        """
        Loads fMRI (BOLD) time series, Structural Connectivity matrices, and distance (delay) matrices, and splits BOLD time series into chunks
        """
        self.fmri_filename = fmri_filename
        self.sc_path = sc_path # Log-transformed and normalised
        self.distance_matrix_path = distance_matrix_path # distance matrix (Euclidean distance in mm)
        self.chunk_length = chunk_length
        self.all_bold = []      # list of BOLD arrays, each shape (node_size, num_TRs)
        self.all_SC = []        # list of SC matrices, each shape (node_size, node_size)
        self.all_distances = [] # list of dist_matrix, each shape (node_size, node_size)
        self.bold_chunks = []   # list of dicts: {'subject': int, 'bold': array (node_size, chunk_length)}
        self.distance_matrix = None
        self.num_subjects = 0
        self.csv_sc = self._load_csv_sc()

        self._load_data()

    def get_node_size(self):
        if len(self.all_SC) == 0: 
            return 0
        return self.all_SC[0].shape[0]
    
    def get_distance_matrix(self):
        return torch.tensor(self.distance_matrix, dtype=torch.float32, device=DEVICE)

    def _load_data(self):
        fmri_mat = scipy.io.loadmat(self.fmri_filename)
        bold_data = fmri_mat["BOLD_timeseries_HCP"]    # shape (100, 1)
        self.num_subjects = bold_data.shape[0]
        
        for subject in range(self.num_subjects):
            bold_subject = bold_data[subject, 0]  # shape (100, 1189)
            self.all_bold.append(bold_subject)
            
            # SC pre-processing: symmetric, max-normalised
            sc_path = os.path.join(self.sc_path, f"sc_norm_subj{subject}.npy")
            sc_norm = np.load(sc_path)
            self.all_SC.append(sc_norm)

        self.distance_matrix = np.load(self.distance_matrix_path)

        print(f"[DataLoader] Loaded {self.num_subjects} subjects.")

    def _split_into_chunks(self):
        self.bold_chunks = []
        for subject, bold_subject in enumerate(self.all_bold):
            num_TRs = bold_subject.shape[1]
            num_chunks = num_TRs // self.chunk_length
            for i in range(num_chunks):
                chunk = bold_subject[:, i*self.chunk_length:(i+1)*self.chunk_length]
                self.bold_chunks.append({"subject": subject, "bold": chunk})
        print(f"[DataLoader] Created {len(self.bold_chunks)} chunks (chunk length = {self.chunk_length}).")

    def load_all_ground_truth(self, builder):
        """
        Loads all chunked ground truth data as PyTorchGeometric.Data for discriminator training 
        Parameters:
            builder: GraphBuilder, defined inside discriminator package
        """
        gt_graphs = []
        for subject in range(len(self.all_bold)):
            sc = self.all_SC[subject]
            gt_graphs.extend(
                [builder.build_graph(torch.tensor(chunk["bold"], dtype=torch.float32, device=DEVICE), torch.tensor(sc, dtype=torch.float32, device=builder.device), label=1.0) for chunk in self.bold_chunks if chunk["subject"] == subject]
            )

        return gt_graphs

    def sample_minibatch(self, batch_size: int):
        sampled = random.sample(self.bold_chunks, batch_size)
        batched_bold = []
        batched_SC = []
        batch_subjects = []

        for batch_element in sampled:
            batched_bold.append(batch_element["bold"]) # (node_size, chunk_length)
            subject = batch_element["subject"]
            batch_subjects.append(subject)

            sc_norm = self.all_SC[subject]  #self.csv_sc 
            batched_SC.append(sc_norm)

        # Stack BOLD
        batched_bold = np.stack(batched_bold, axis=-1) # (node_size, chunk_length, batch_size)
        batched_bold = torch.tensor(batched_bold, dtype=torch.float32, device=DEVICE)

        # Stack batched SC
        batched_SC = np.stack(batched_SC, axis=0)
        batched_SC = torch.tensor(batched_SC, dtype=torch.float32, device=DEVICE)

        batch_subjects = torch.tensor(batch_subjects, dtype=torch.int32, device=DEVICE)

        return batched_bold, batched_SC, batch_subjects



    def _load_csv_sc(self, csv_path: str = "/vol/bitbucket/ank121/fyp/HCP Data/fiber_consensus.csv", verbose=False):
        sc = np.loadtxt(csv_path, delimiter=",")
        sc_norm = sc / sc.max()                         # global-max scaling
        row_sum_csv = sc_norm.sum(axis=1)

        if verbose:
            print(f"[DataLoader] Consensus SC (CSV):")
            print(f"  nodes            : {sc_norm.shape[0]}")
            print(f"  row-sum  min     : {row_sum_csv.min():.4f}")
            print(f"  row-sum  mean    : {row_sum_csv.mean():.4f}")
            print(f"  row-sum  max     : {row_sum_csv.max():.4f}")
            print(f"  row-sum  std     : {row_sum_csv.std():.4f}")

        return sc_norm
    
    def get_subject_connectome(self, subj: int, norm=True):
        """Return SC as torch.float32 on the correct device."""
        sc_np = self.all_SC[subj]
        if norm: 
            sc = np.log1p(sc_np) / np.linalg.norm(np.log1p(sc_np))
            return torch.tensor(sc, dtype=torch.float32, device=DEVICE)
        return torch.tensor(sc_np, dtype=torch.float32, device=DEVICE)

    def bold_chunk_to_fc(self, bold_chunk: np.ndarray) -> np.ndarray:
        """ Turn (N, T) chunk into (N,N) Pearson-FC """
        return torch.tensor(bold_to_fc(bold_chunk), dtype=torch.float32)

    def iter_chunks_per_subject(self, subject: int):
        """ Yield successive (bold_chunk, fc_chunk) from self.bold_chunks """
        for ch in self.bold_chunks:
            if ch["subject"] == subject:
                yield ch["bold"], self.bold_chunk_to_fc(ch["bold"])

