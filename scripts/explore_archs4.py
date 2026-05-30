import archs4py as a4
import h5py
import pandas as pd

if __name__ == "__main__":
    disease_annotations = pd.read_csv("data/disign_atlas/control_case_gsms.csv")
    gsms = disease_annotations["gsm"].unique()
    archs4_dataset_path = "data/archs4/human_gene_v2.latest.h5"

    with h5py.File(archs4_dataset_path, 'r') as f:
        str_fields = [k for k in f['meta/samples'].keys() if f['meta/samples'][k].dtype.kind in ('S', 'U', 'O')]
        geo = f['meta/samples/geo_accession'][:].astype(str)
        library_strategy = f['meta/samples/library_strategy'][:].astype(str)
        aligned = f['meta/samples/alignedreads'][:]

    mask = library_strategy == 'RNA-Seq'
    rnaseq_geo = geo[mask]
    rnaseq_aligned = aligned[mask]

    meta = a4.meta.samples(archs4_dataset_path, rnaseq_geo, str_fields)
    meta['alignedreads'] = rnaseq_aligned
    print(meta.shape)
    print(meta.head())


