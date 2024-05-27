#!/bin/bash

wget -O data/prostate/P1000_final_analysis_set_cross_important_only.csv --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1EqQ-_34Q404E0CZfbztz7l8dhqE3zmOI'
wget -O data/prostate/P1000_data_CNA_paper.csv --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1D5v7ORm1qLeAPc5IaqzagYQJsccpAbcZ'
wget -P data/prostate https://s3-us-west-2.amazonaws.com/humanbase/networks/prostate_gland.gz
wget -O data/brain/response.csv --no-check-certificate -r 'https://drive.google.com/file/d/1rfwsBvuEHKaOlJ9NS89ENulsZeriUjhL/view?usp=drive_link'
wget -O data/brain/mut.csv --no-check-certificate -r 'https://drive.google.com/file/d/1wBnI9GryDYV9n5h6x5tnvQ28qSCxESDI/view?usp=drive_link'
wget -O data/brain/cnv.csv --no-check-certificate -r 'https://drive.google.com/file/d/1CXEJWiNJdIQHEsqpWih-Xk0EfZl8RXSV/view?usp=drive_link'
wget -O data/brain/brain_vector.pkl --no-check-certificate -r 'https://drive.google.com/file/d/1X95iF1AAG8hvZL2TTl1TmDEr4uuLWzMm/view?usp=drive_link'
