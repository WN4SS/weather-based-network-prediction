# Pre-Processing provided dataset
## Using given dataset
Run the following lines to replicate the dataset used in the paper
~~~zsh
python3 preprocessing_Buffalo.py --kpi_filepath "data/KPI_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --weather_filepath "data/Weather_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --remove_outliers  False --last_timestamp "2024-04-25 19:00:00" --savename "Buffalo_80"

python3 preprocessing_scripts/preprocessing_Buffalo.py --kpi_filepath "FIAR Database-selected/KPI_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --weather_filepath "FIAR Database-selected/Weather_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --remove_outliers  True --first_timestamp "2024-04-25 20:00:00" --savename "Buffalo_80"

python3 preprocessing_scripts/preprocessing_Buffalo.py --kpi_filepath "FIAR Database-selected/KPI_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --weather_filepath "FIAR Database-selected/Weather_Buffalo_80_from_06_20_2022_to_05_31_2024.csv" --remove_outliers  False --first_timestamp "2024-04-25 20:00:00" --savename "Buffalo_80"

python3 preprocessing_scripts/preprocessing_Elmira.py --kpi_filepath "FIAR Database-selected/KPI_Elmira.csv" --weather_filepath "FIAR Database-selected/Weather_Elmira.csv" --remove_outliers  False --savename "Elmira"
~~~
## Preparing your own dataset
Use the following command to view all possible arguments. 
~~~zsh
python3 preprocessing.py -h
~~~
At minimum, you will need to provide the file paths for KPI and Weather data, and the output file save name

