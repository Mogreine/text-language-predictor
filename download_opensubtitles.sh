#!/usr/bin/env bash
wget https://falca-bucket.s3.eu-central-1.amazonaws.com/open_subtitles.zip
mkdir data/open_subtitles
unzip open_subtitles.zip -d data/open_subtitles
rm open_subtitles.zip
