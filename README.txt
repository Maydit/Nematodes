To activate the pip environment:

python3.8 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

To see help for running the python script:

python wormcounter.py -h

Example for running the script:

python wormcounter.py ../images/example_image.tiff ../results/example_count.csv

To run on cluster:

sbatch run.s FILE1 FILE2 (-vv or -v or nothing)


Sidenote: as a consequence of untarring this WILL NOT work with files inside a .tar.gz that have ._ in their name.
