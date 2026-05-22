import re
import h5py
import numpy as np
import pandas as pd
import archs4py as a4
from tqdm import tqdm

ARCHS4_PATH = "data/archs4/human_gene_v2.latest.h5"

MUST_HAVE = {
    "TNF", "IL6", "IL1B", "IFNG",
    "CXCL8", "CXCL10", "CCL2",
    "CCR7", "CXCR4", "IL6R",
    "TLR4", "NLRP3", "MYD88",
    "HLA-DRA", "B2M", "TAP1",
    "STAT1", "NFKB1", "RELA",
}

HOUSEKEEPING = {
    "ACTB", "GAPDH", "HPRT1", "PPIA", "PGK1", "GUSB",
    "TFRC", "YWHAZ", "SDHA", "UBC", "RPLP0", "RPL13A",
    "RPS18", "TBP", "HMBS", "IPO8",
}

HGNC_GROUP_PATTERNS = {
    # immune / inflammation
    "cytokines_growth_factors": [
        r"cytokine", r"interleukin", r"interferon", r"growth factor",
        r"TNF", r"VEGF", r"Wnt family", r"Transforming growth factor",
        r"Hedgehog", r"Notch", r"EGF.CFC",
    ],
    "chemokines": [r"chemokine"],
    "cytokine_chemokine_receptors": [
        r"chemokine receptor", r"Interleukin receptor", r"Interferon receptor",
        r"TNF receptor", r"cytokine receptor",
    ],
    "antigen_presentation": [r"Histocompatibility complex", r"Minor histocompatibility"],
    "innate_sensing": [r"Toll like receptor", r"NLR family", r"RIG-I"],

    # regulation / signaling
    "transcription_factors": [
        r"transcription factor", r"homeobox",
        r"forkhead", r"bHLH", r"bZIP", r"nuclear factor",
        # zinc finger: C2H2-type and KZNFs are TFs; exclude RING/TRIM (E3 ligases)
        r"zinc finger.*C2H2", r"zinc finger.*KRAB", r"zinc finger.*Krüppel",
        r"zinc finger.*homeodomain", r"zinc finger.*C4",
    ],
    "protein_kinases": [
        # exclude metabolic kinases (pyruvate, adenylate, etc.) and lipid kinases
        r"protein kinase", r"tyrosine kinase", r"serine.threonine kinase",
        r"receptor.*kinase", r"non-receptor.*kinase",
        r"MAP kinase", r"Aurora kinase", r"polo.like kinase",
        r"checkpoint kinase", r"cyclin.dependent kinase",
    ],
    "phosphatases": [
        # exclude lipid/sugar phosphatases
        r"protein phosphatase", r"tyrosine phosphatase",
        r"phosphoprotein phosphatase", r"dual specificity phosphatase",
    ],
    "signal_transduction_adaptors": [
        # exclude clathrin/endocytic adaptors
        r"signaling adaptor", r"signal.*adaptor", r"scaffold protein",
        r"GTPase activating protein", r"guanine nucleotide exchange factor",
        r"Rho GTP", r"Ras.*(GTP|onco)",
        r"MAPK", r"MAP kinase",
    ],
    "nuclear_receptors": [r"nuclear receptor subfamily", r"Nuclear receptor subfamily"],

    # core cellular programs
    "cell_cycle": [r"cyclin", r"CDK", r"cell cycle regulator", r"checkpoint kinase"],
    "dna_damage_repair": [r"DNA repair", r"nucleotide excision repair", r"mismatch repair", r"DNA damage"],
    "apoptosis_cell_death": [r"caspase", r"BCL2", r"death.*domain", r"death.*effector", r"apoptosis"],
    "stress_response": [r"heat shock", r"chaperone"],

    # metabolism / organelles
    "mitochondria_oxidative_phosphorylation": [
        r"mitochondri", r"respiratory chain", r"ATP synthase subunit",
        r"cytochrome c oxidase", r"succinate dehydrogenase",
    ],
    "ribosome_translation": [
        r"ribosomal protein", r"ribosomal subunit", r"ribosomal biogenesis",
        r"translation initiation", r"translation.*factor",
    ],
    "proteasome_ubiquitin": [r"proteasome", r"ubiquitin"],
    "autophagy_lysosome": [r"autophagy", r"lysosom"],

    # tissue / structure
    "ecm_adhesion": [r"collagen", r"integrin", r"laminin", r"fibronectin", r"adhesion molecule"],
    "cytoskeleton_motility": [r"Actins", r"actinin", r"myosin", r"tubulin", r"kinesin", r"dynein"],
    "epithelial_markers": [r"keratin"],
    "endothelial_markers": [r"endothelin"],

    # receptors / transport / membrane biology
    "gpcrs": [r"G protein.coupled receptor", r"olfactory receptor"],
    "ion_channels": [
        r"ion channel", r"potassium.*channel", r"sodium.*channel",
        r"calcium channel", r"chloride channel",
    ],
    "solute_carriers": [r"Solute carrier family"],
    "transporters": [
        r"ATP binding cassette", r"ABC transporter",
        r"solute carrier", r"major facilitator",
        r"aquaporin", r"ATPase",
    ],
}

gene_group_quota = {
    'l1000': 500,

    # immune / inflammation
    "cytokines_growth_factors": 40,
    "chemokines": 25,
    "cytokine_chemokine_receptors": 45,
    "c8_immune_sig": 45,
    "antigen_presentation": 20,
    "innate_sensing": 25,
    "interferon_response": 25,

    # regulation / signaling
    "transcription_factors": 90,
    "protein_kinases": 70,
    "phosphatases": 25,
    "signal_transduction_adaptors": 35,
    "nuclear_receptors": 20,

    # core cellular programs
    "cell_cycle": 45,
    "dna_damage_repair": 30,
    "apoptosis_cell_death": 30,
    "stress_response": 25,

    # metabolism / organelles
    "metabolism": 50,
    "mitochondria_oxidative_phosphorylation": 35,
    "ribosome_translation": 25,
    "proteasome_ubiquitin": 25,
    "autophagy_lysosome": 20,

    # tissue / structure
    "ecm_adhesion": 40,
    "cytoskeleton_motility": 30,
    "c8_epithelial_sig": 20,
    "c8_endothelial_sig": 20,
    "c8_fibroblast_stromal_sig": 20,

    # receptors / transport / membrane biology
    "gpcrs": 25,
    "ion_channels": 20,
    "transporters": 35,
    "solute_carriers": 25,

    # technical anchors — quota matches HOUSEKEEPING set size
    "housekeeping": 16,
}

HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)

MSIGDB_RELEASE_BASE = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"

MSIGDB_PATTERNS = {
    # Hallmark (H)
    "cell_cycle":                          ("H", r"G2M_CHECKPOINT|E2F_TARGETS|MITOTIC_SPINDLE"),
    "dna_damage_repair":                   ("H", r"DNA_REPAIR|P53_PATHWAY|UV_RESPONSE"),
    "apoptosis_cell_death":                ("H", r"APOPTOSIS"),
    "interferon_response":                 ("H", r"INTERFERON"),
    "metabolism":                          ("H", r"METABOLISM|GLYCOLYSIS|FATTY_ACID|OXIDATIVE_PHOSPHORYLATION|CHOLESTEROL|PEROXISOME|HEME_METABOLISM|XENOBIOTIC"),
    "mitochondria_oxidative_phosphorylation": ("H", r"OXIDATIVE_PHOSPHORYLATION"),
    "ecm_adhesion":                        ("H", r"EPITHELIAL_MESENCHYMAL_TRANSITION|COAGULATION|ANGIOGENESIS|APICAL_JUNCTION"),
    # C8 cell-type signatures — labels prefixed c8_ to signal these are broad signature memberships,
    # not cell-type-specific markers (gene counts will be high; that is expected)
    "c8_immune_sig":                       ("C8", r"T_CELL|B_CELL|NK_CELL|MACROPHAGE|DENDRITIC|MONOCYTE|NEUTROPHIL|MAST_CELL|PLASMA_CELL"),
    "c8_epithelial_sig":                   ("C8", r"EPITHELIAL"),
    "c8_endothelial_sig":                  ("C8", r"ENDOTHELIAL"),
    "c8_fibroblast_stromal_sig":           ("C8", r"FIBROBLAST|STROMAL"),
}


# Genes excluded from the panel regardless of score.
# Non-symbol IDs, pseudogene artifacts, and MT transcripts (tend to capture
# RNA quality / tissue composition rather than biological signal).
EXCLUDE_GENES = {
    "RPL23AP42",
    # sex-chromosome markers — intentionally excluded to avoid sex composition
    # dominating the embedding space; tag separately if needed downstream
    "XIST", "DDX3Y", "KDM5D", "RPS4Y1", "USP9Y", "UTY",
}
EXCLUDE_PREFIXES = ("MT-",)


# QC thresholds
MIN_ALIGNED_READS = 1_000_000
MAX_SC_PROBABILITY = 0.5
MIN_LOG_EXPRESSION_THRESHOLD = 1.0
MIN_GENES_EXPRESSED = 200


def get_l1000_genes() -> np.ndarray:
    """Fetch LINCS L1000 landmark gene symbols from GEO.

    Returns:
        Array of gene symbols for the 978 landmark genes (pr_is_lm == 1).
    """
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/"
        "GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz"
    )
    gene_info = pd.read_csv(url, sep="\t")
    return gene_info.loc[gene_info["pr_is_lm"] == 1, "pr_gene_symbol"].dropna().unique()


def hgnc_group_match(hgnc: pd.DataFrame, patterns: list[str]) -> set[str]:
    """Return HGNC gene symbols whose gene_group matches any of the given patterns.

    The HGNC gene_group column is pipe-delimited; each gene may belong to
    multiple groups. The function explodes the column before matching so that
    a gene is included if *any* of its groups matches.

    Args:
        hgnc: HGNC complete set DataFrame with at least 'symbol' and 'gene_group' columns.
        patterns: List of regex fragments joined with | and matched case-insensitively.

    Returns:
        Set of upper-cased gene symbols.
    """
    regex = re.compile("|".join(patterns), re.I)
    exploded = hgnc[["symbol", "gene_group"]].dropna()
    exploded = exploded.assign(gene_group=exploded["gene_group"].str.split("|")).explode("gene_group")
    mask = exploded["gene_group"].str.strip().str.contains(regex, na=False)
    return set(exploded.loc[mask, "symbol"].str.upper())


def _resolve_msigdb_release() -> str:
    """Probe Broad servers to find the latest available MSigDB release version.

    Tries year.minor.Hs combinations in descending order (2026.2 → 2022.1),
    returning the first version whose Hallmark GMT file returns HTTP 200.

    Returns:
        Version string such as '2025.1.Hs'.

    Raises:
        RuntimeError: If no reachable release is found in the probed range.
    """
    import requests
    import itertools
    for year, minor in itertools.product(range(2026, 2022, -1), (2, 1)):
        version = f"{year}.{minor}.Hs"
        url = f"{MSIGDB_RELEASE_BASE}/{version}/h.all.v{version}.symbols.gmt"
        status = requests.head(url, timeout=10, allow_redirects=True).status_code
        print(f"  {url} -> {status}")
        if status == 200:
            return version
    raise RuntimeError("Could not resolve latest MSigDB release")


def msigdb_group_match(collection: str, set_name_pattern: str, version: str) -> set[str]:
    """Fetch a MSigDB GMT file and return genes from sets matching the name pattern.

    Args:
        collection: MSigDB collection identifier (e.g. 'H', 'C8').
        set_name_pattern: Regex applied to gene set names (first column of GMT).
        version: MSigDB release version string (e.g. '2025.1.Hs').

    Returns:
        Union of all gene symbols from matching gene sets.

    Raises:
        requests.HTTPError: If the GMT file cannot be fetched.
    """
    import requests
    url = f"{MSIGDB_RELEASE_BASE}/{version}/{collection.lower()}.all.v{version}.symbols.gmt"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    regex = re.compile(set_name_pattern, re.I)
    genes: set[str] = set()
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        if regex.search(parts[0]):
            genes.update(parts[2:])
    return genes


def load_samples(archs4_path: str) -> pd.DataFrame:
    """Load sample-level metadata from an ARCHS4 HDF5 file.

    Args:
        archs4_path: Path to the ARCHS4 .h5 file.

    Returns:
        DataFrame with columns: geo_accession, library_strategy,
        alignedreads, singlecellprobability.
    """
    with h5py.File(archs4_path, "r") as f:
        return pd.DataFrame({
            "geo_accession": f["meta/samples/geo_accession"][:].astype(str),
            "library_strategy": f["meta/samples/library_strategy"][:].astype(str),
            "alignedreads": f["meta/samples/alignedreads"][:],
            "singlecellprobability": f["meta/samples/singlecellprobability"][:],
        })


def qc_samples(meta: pd.DataFrame) -> pd.DataFrame:
    """Filter samples to bulk RNA-seq with sufficient depth and low single-cell probability.

    Args:
        meta: Sample metadata DataFrame from load_samples().

    Returns:
        Filtered DataFrame with reset index.
    """
    n = len(meta)
    meta = meta[meta["library_strategy"] == "RNA-Seq"]
    print(f"RNA-Seq filter: {len(meta)}/{n}")

    n = len(meta)
    meta = meta[meta["singlecellprobability"] < MAX_SC_PROBABILITY]
    print(f"SC filter (prob<{MAX_SC_PROBABILITY}): {len(meta)}/{n}")

    n = len(meta)
    meta = meta[meta["alignedreads"] >= MIN_ALIGNED_READS]
    print(f"Aligned reads filter (>={MIN_ALIGNED_READS:,}): {len(meta)}/{n}")

    return meta.reset_index(drop=True)


def log_transform_and_filter_samples(expr: pd.DataFrame) -> np.ndarray:
    """Apply log1p and drop samples with too few expressed genes.

    Operates on a genes × samples matrix. Samples where fewer than
    MIN_GENES_EXPRESSED genes exceed MIN_LOG_EXPRESSION_THRESHOLD are removed
    to exclude low-quality or near-empty libraries.

    Args:
        expr: Raw count DataFrame with shape (n_genes, n_samples).

    Returns:
        log1p-transformed float32 array of shape (n_genes, n_passing_samples).
    """
    log_expr = np.log1p(expr.values.astype(np.float32))
    sample_mask = (log_expr > MIN_LOG_EXPRESSION_THRESHOLD).sum(axis=0) >= MIN_GENES_EXPRESSED
    return log_expr[:, sample_mask]


def filter_and_rank_genes(stats: pd.DataFrame) -> pd.DataFrame:
    """Remove lowly expressed / rare genes and rank remainder by a composite score.

    Score is a min-max normalised weighted sum of three metrics:
        robust_variance (0.5) + prevalence (0.3) + mean_log_expression (0.2).
    Robust variance (IQR-based σ²) is weighted highest to favour
    informative genes over constitutively expressed housekeeping genes.

    Args:
        stats: DataFrame with columns gene, prevalence, mean_log_expression,
            robust_variance — as produced by compute_gene_stats().

    Returns:
        Filtered and scored DataFrame sorted descending by score, index reset.
    """
    stats = stats[(stats["prevalence"] >= 0.05) & (stats["mean_log_expression"] >= 0.1)].copy()
    print(f"Genes passing QC: {len(stats)}")

    weights = {"robust_variance": 0.5, "prevalence": 0.3, "mean_log_expression": 0.2}
    for col in weights:
        col_min, col_max = stats[col].min(), stats[col].max()
        stats[f"{col}_norm"] = (stats[col] - col_min) / (col_max - col_min)
    stats["score"] = sum(stats[f"{col}_norm"] * w for col, w in weights.items())
    stats = stats.drop(columns=[f"{col}_norm" for col in weights])
    return stats.sort_values("score", ascending=False).reset_index(drop=True)


def compute_gene_stats(genes: np.ndarray, expr: pd.DataFrame) -> pd.DataFrame:
    """Compute per-gene expression statistics over a genes × samples matrix.

    Args:
        genes: Gene symbol array aligned to expr rows.
        expr: Raw count DataFrame with shape (n_genes, n_samples).

    Returns:
        DataFrame with columns: gene, prevalence, mean_log_expression,
        robust_variance (IQR-based σ²).
    """
    assert expr.shape[0] == len(genes), "Expected genes x samples"
    log_expr = log_transform_and_filter_samples(expr)
    expressed = log_expr > MIN_LOG_EXPRESSION_THRESHOLD
    q75 = np.percentile(log_expr, 75, axis=1)
    q25 = np.percentile(log_expr, 25, axis=1)

    return pd.DataFrame({
        "gene": genes,
        "prevalence": expressed.mean(axis=1),
        "mean_log_expression": log_expr.mean(axis=1),
        "robust_variance": ((q75 - q25) / 1.35) ** 2,
    })


def can_remove_gene(gene: str, panel: set, group_genes: dict, quotas: dict) -> bool:
    """Return True if removing a gene from the panel would not violate any group quota.

    Args:
        gene: Candidate gene to remove.
        panel: Current panel set.
        group_genes: Mapping of group name → set of member genes.
        quotas: Mapping of group name → minimum required panel members.

    Returns:
        True if removal is safe, False if it would drop any group below its quota.
    """
    for group, quota in quotas.items():
        if group == "l1000":
            continue
        if gene in group_genes.get(group, set()):
            if len((panel - {gene}) & group_genes.get(group, set())) < quota:
                return False
    return True


if __name__ == "__main__":
    l1000_genes = get_l1000_genes()
    print(f"L1000 landmark genes: {len(l1000_genes)}")
    meta = load_samples(ARCHS4_PATH)
    meta = qc_samples(meta)
    print(f"QC-passing samples: {len(meta)}")

    meta = meta.sample(min(20000, len(meta)), random_state=42)
    expr = a4.data.samples(ARCHS4_PATH, meta["geo_accession"].tolist(), silent=False)

    print("Aggregating duplicate genes...")
    expr = a4.utils.aggregate_duplicate_genes(expr)

    hgnc_complete_set = pd.read_csv(HGNC_COMPLETE_SET_URL, sep="\t", low_memory=False)
    protein_coding = set(
        hgnc_complete_set.loc[
            hgnc_complete_set["locus_group"] == "protein-coding gene", "symbol"
        ].str.upper().dropna()
    )
    print(f"Protein-coding genes in HGNC: {len(protein_coding)}")

    # symbol → ensembl_id (one-to-one for most genes)
    symbol_to_ensembl: dict[str, str] = (
        hgnc_complete_set[["symbol", "ensembl_gene_id"]]
        .dropna()
        .set_index("symbol")["ensembl_gene_id"]
        .to_dict()
    )
    # ensembl_id → all symbols (one-to-many for duplicate/retired entries)
    ensembl_to_symbols: dict[str, list[str]] = (
        hgnc_complete_set[["symbol", "ensembl_gene_id"]]
        .dropna()
        .groupby("ensembl_gene_id")["symbol"]
        .apply(list)
        .to_dict()
    )

    raw_stats = compute_gene_stats(expr.index.to_numpy(), expr)
    n_before = len(raw_stats)
    sorted_gene_stats = filter_and_rank_genes(
        raw_stats[
            raw_stats["gene"].isin(protein_coding) &
            ~raw_stats["gene"].isin(EXCLUDE_GENES) &
            ~raw_stats["gene"].str.startswith(EXCLUDE_PREFIXES)
        ]
    )
    print(f"Protein-coding + exclusion filter: removed {n_before - len(sorted_gene_stats)} genes ({len(sorted_gene_stats)} remain)")

    group_genes: dict[str, set[str]] = {}
    for group, patterns in tqdm(HGNC_GROUP_PATTERNS.items(), desc="Fetching HGNC groups"):
        group_genes[group] = hgnc_group_match(hgnc_complete_set, patterns)

    # housekeeping from manual list, not HGNC regex
    group_genes["housekeeping"] = HOUSEKEEPING

    msigdb_version = _resolve_msigdb_release()
    print(f"MSigDB release: {msigdb_version}")
    for group, (collection, pattern) in tqdm(MSIGDB_PATTERNS.items(), desc="Fetching MSigDB groups"):
        group_genes[group] = (
            group_genes.get(group, set())
            | msigdb_group_match(collection, pattern, msigdb_version)
        )

    score_map = sorted_gene_stats.set_index("gene")["score"].to_dict()

    # seed panel with must-have genes that passed QC
    panel = set(MUST_HAVE) & set(sorted_gene_stats["gene"])
    print(f"Must-have seeds: {len(panel)}")

    selected: dict[str, list[str]] = {}

    # 1. L1000 backbone, capped at quota
    selected["l1000"] = (
        sorted_gene_stats[sorted_gene_stats["gene"].isin(l1000_genes)]
        .head(gene_group_quota["l1000"])["gene"]
        .tolist()
    )
    panel.update(selected["l1000"])
    print(f"After L1000: {len(panel)}")

    # 2. Enforce group quotas as minimums
    for group, quota in gene_group_quota.items():
        if group == "l1000":
            continue

        current_in_group = panel & group_genes.get(group, set())
        needed = max(0, quota - len(current_in_group))

        candidates = sorted_gene_stats[
            sorted_gene_stats["gene"].isin(group_genes.get(group, set())) &
            ~sorted_gene_stats["gene"].isin(panel)
        ]
        added = candidates.head(needed)["gene"].tolist()
        selected[group] = added
        panel.update(added)
        total_in_group = len(panel & group_genes.get(group, set()))
        print(f"{group}: added {len(added)}, total in panel {total_in_group}/{quota}")

    # 3. Fill to 1000 from top-scored remaining genes
    if len(panel) < 1000:
        filler = sorted_gene_stats[~sorted_gene_stats["gene"].isin(panel)].head(1000 - len(panel))["gene"].tolist()
        selected["global_fill"] = filler
        panel.update(filler)
        print(f"Global fill: added {len(filler)}, panel now {len(panel)}")

    # 4. Trim to 1000, protecting only must-haves; skip genes whose removal would break a quota
    if len(panel) > 1000:
        protected = MUST_HAVE & set(sorted_gene_stats["gene"])
        removable = sorted(panel - protected, key=lambda g: score_map.get(g, 0))
        removed = 0
        while len(panel) > 1000 and removable:
            gene = removable.pop(0)
            if can_remove_gene(gene, panel, group_genes, gene_group_quota):
                panel.remove(gene)
                removed += 1
        print(f"After trim: removed {removed}, panel now {len(panel)}")

    print(f"\nTotal unique genes selected: {len(panel)}")

    # build output from panel so every gene (including must-have-only) is represented
    gene_to_groups: dict[str, list[str]] = {g: [] for g in panel}
    for group, genes_list in selected.items():
        for g in genes_list:
            if g in panel:
                gene_to_groups[g].append(group)
    for g in panel:
        if g in MUST_HAVE:
            gene_to_groups[g].append("must_have")
        for group, genes in group_genes.items():
            if g in genes:
                gene_to_groups[g].append(group)

    def _ensembl_id(gene: str) -> str:
        # ENSG-named genes in ARCHS4 are already Ensembl IDs
        return gene if gene.startswith("ENSG") else symbol_to_ensembl.get(gene, "")

    selected_genes_df = pd.DataFrame([
        {
            "ensembl_id": _ensembl_id(g),
            "gene_symbols": ";".join(ensembl_to_symbols.get(_ensembl_id(g), [g])),
            "groups": ";".join(sorted(set(gene_to_groups[g]))),
            "score": score_map.get(g, np.nan),
            "is_must_have": g in MUST_HAVE,
            "is_l1000": g in set(l1000_genes),
        }
        for g in sorted(panel)
    ])
    import os
    out_path = "data/genes/selected_genes.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    selected_genes_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
