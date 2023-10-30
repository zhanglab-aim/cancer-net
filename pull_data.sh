#!/bin/bash

wget -O data/prostate/P1000_final_analysis_set_cross_important_only.csv --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1EqQ-_34Q404E0CZfbztz7l8dhqE3zmOI'
wget -O data/prostate/P1000_data_CNA_paper.csv --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1D5v7ORm1qLeAPc5IaqzagYQJsccpAbcZ'
wget -P data/prostate https://s3-us-west-2.amazonaws.com/humanbase/networks/prostate_gland.gz


