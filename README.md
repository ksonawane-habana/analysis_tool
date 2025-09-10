# analysis_tool
A tool to analyze tpc-fuser feature performance results.

# Prep
Run models tests commands:

ASIC 1 : (debug data with detailed subgraph info as well)
```
ENABLE_EXPERIMENTAL_FLAGS=true ENABLE_FUSION_BEFORE_NORM=false FUSER_DEBUG_DATA=1 run_models_tests -l d -n withoutfeature -c gaudi3 -w g3_promo_models --jobs promotion --graphs --save_post_graph
ENABLE_EXPERIMENTAL_FLAGS=true ENABLE_FUSION_BEFORE_NORM=true FUSER_DEBUG_DATA=1 run_models_tests -l d -n feature -c gaudi3 -w g3_promo_models --jobs promotion --ref_name withoutfeature --graphs --save_post_graph
```

ASIC 2 : (basic report generation)
```
ENABLE_EXPERIMENTAL_FLAGS=true ENABLE_FUSION_BEFORE_NORM=false FUSER_DEBUG_DATA=1 run_models_tests -l d -n withoutfeature -c gaudi3 -w g3_promo_models_full --jobs promotion
ENABLE_EXPERIMENTAL_FLAGS=true ENABLE_FUSION_BEFORE_NORM=true FUSER_DEBUG_DATA=1 run_models_tests -l d -n feature -c gaudi3 -w g3_promo_models_full --jobs promotion --ref_name withoutfeature
```
## Expected Directory structure:
```
(venv) ksonawan@ksonawan-vm-u24:fusion_before_norm $ tree -d -L 2
.
├── asic1
│   └── g3_promo_models
└── asic2
    └── g3_promo_models_full

5 directories
```
# Run:
```
cd workspace
python -m venv venv
source venv/bin/activate
pip install Django numpy pandas

FEATURE_DIR=<Feature base Dir> python manage.py runserver
```
