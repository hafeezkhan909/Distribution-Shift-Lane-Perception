# predictor/forms.py

from django import forms

class ExperimentForm(forms.Form):
    """
    A Django Form that maps directly to the argparse arguments
    from the experiment script.
    """
    
    # --- Dataset Arguments ---
    source = forms.CharField(
        label="Source Dataset", 
        initial="Curvelanes",
        help_text="Source dataset name"
    )
    target = forms.CharField(
        label="Target Dataset", 
        initial="Curvelanes",
        help_text="Target dataset name"
    )
    src_split = forms.CharField(
        label="Source Split", 
        initial="train",
        help_text="Source dataset split (e.g., 'train', 'val')"
    )
    tgt_split = forms.CharField(
        label="Target Split", 
        initial="train",
        help_text="Target dataset split (e.g., 'train', 'val')"
    )

    # --- Sampling Arguments ---
    src_samples = forms.IntegerField(
        label="Source Samples", 
        initial=1000,
        help_text="Number of samples for the source reference"
    )
    tgt_samples = forms.IntegerField(
        label="Target Samples", 
        initial=100,
        help_text="Number of samples for the target test"
    )
    block_idx = forms.IntegerField(
        label="Block Index", 
        initial=0,
        help_text="Block index for chunked source loading"
    )

    # --- Model & MMD Test Arguments ---
    batch_size = forms.IntegerField(
        label="Batch Size", 
        initial=16,
        help_text="Batch size for feature extraction"
    )
    image_size = forms.IntegerField(
        label="Image Size", 
        initial=512,
        help_text="Image resize dimension"
    )
    num_calib = forms.IntegerField(
        label="Number of Calibrations", 
        initial=100,
        help_text="Number of calibration runs for null distribution"
    )
    alpha = forms.FloatField(
        label="Alpha (Significance Level)", 
        initial=0.05,
        help_text="Significance level for the test"
    )
    
    # --- Reproducibility ---
    seed_base = forms.IntegerField(
        label="Base Seed", 
        initial=42,
        help_text="Base seed for random sampling"
    )
