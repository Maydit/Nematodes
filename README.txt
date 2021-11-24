To activate the pip environment:

python -m venv env
source ./env/Scripts/Activate
python -m pip install -r requirements.txt

To see help for running the python script:

python wormcounter.py -h

Example for running the script:

python wormcounter.py ../images/example_image.tiff ../results/example_count.csv




Sidenote: as a consequence of untarring this WILL NOT work with files inside a .tar.gz that have ._ in their name.