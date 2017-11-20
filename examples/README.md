Hi.  The examples directory could use its own readme...  ...but for now the code *is* the README.   
Just look at the comments in the source before you try to run anything. ;-)

Few quick thoughts:

The two *_setup* files are what you want to run.  They will grab data from elsewhere and provide prompts and instructions. 

**headgames.py** is a GUI for a "live demo" I'm working on, once the network has been trained, but I don't expect it to be easy for anyone else to use (yet).

Sample usage:

    ./headgames.py -n 8 -w /home/me/panotti/weights.hdf5 -d /home/me/panotti/Samples/Class1/

Run './headgames.py --help' for more info
