bash scripts/train.sh 0  --config cfgs/CRN_models/CustomDAPoinTr_SourceOnly.yaml  --log_dir logs/custom_source_only     --seed 42

bash scripts/train.sh 0 \
    --config cfgs/CRN_models/DAPoinTr.yaml \
    --exp_name crn_source_only \
    --seed 42 \