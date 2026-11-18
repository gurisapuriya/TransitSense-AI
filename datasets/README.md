# TransitSense AI - Dataset Setup

## Overview
This directory contains datasets used for training,
testing, and evaluating TransitSense AI.

## Dataset Sources
See sources.csv for dataset URLs, versions,
formats, and licenses.

## Local Setup
1. Download datasets listed in sources.csv.
2. Extract each dataset into datasets/raw/
   using the folder names in sources.csv.
3. Keep original annotations unchanged.
4. Do not upload images or videos to GitHub.

## Dataset Usage
- bus_occupancy: Passenger detection
- bus_seat_availability: Seat availability
- people_counting: Boarding/exiting videos
- seat_occupancy: Occupied/vacant seats
- trash_detection: Litter detection
- water_spills: Floor spill detection

## Important
Raw and processed datasets are excluded from Git.

Dataset class mappings and training splits
will be prepared separately.

Follow each dataset's licensing conditions.
