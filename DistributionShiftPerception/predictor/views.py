# predictor/views.py

from django.shortcuts import render
from .forms import ExperimentForm

# Import your main function from the refactored script
# 'mlproject' should be the name of your project's root folder
# from mlproject.ml_logic.experiment import main as run_experiment_logic

def run_experiment(request):
    # This context dictionary will hold the form or the results
    context = {}

    if request.method == 'POST':
        # Create a form instance and populate it with data from the request
        form = ExperimentForm(request.POST)
        
        if form.is_valid():
            # Form data is valid. Get the cleaned data as a dictionary.
            experiment_args = form.cleaned_data
            
            try:
                # --- THIS IS THE SYNCHRONOUS CALL ---
                # Call your main ML function with all the form arguments.
                # This will BLOCK the website until it's done.
                
                # results = run_experiment_logic(**experiment_args)

                stub_results = {
                    'p_value': 0.01234,  # Fake p-value
                    'alpha': experiment_args.get('alpha'), # Get alpha from the form
                    'is_distribution_shifted': True, # Fake boolean result
                    'source_dataset': experiment_args.get('source'),
                    'target_dataset': experiment_args.get('target'),
                    'source_samples': experiment_args.get('src_samples'),
                    'target_samples': experiment_args.get('tgt_samples'),
                }
                
                # If successful, render the results page
                context['results'] = results
                return render(request, 'predictor/results.html', context)

            except Exception as e:
                # If your script fails, add the error to the form
                # and re-render the experiment page
                form.add_error(None, f"An error occurred during execution: {e}")
                
        # If form is invalid, or an error occurred, 
        # the page will re-render with the errors shown.
        context['form'] = form
        return render(request, 'predictor/run.html', context)

    else:
        # A GET request: create a new, blank form
        form = ExperimentForm()
        context['form'] = form
        return render(request, 'predictor/run.html', context)
