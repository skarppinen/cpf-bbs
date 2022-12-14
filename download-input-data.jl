## A script to download input data. 
include("config.jl");
using InfoZIP 
import Downloads

# URL where input data can be downloaded from.
input_data_url = "https://nextcloud.jyu.fi/index.php/s/d8WP6gGtyJaZDAM/download";
tempfile = joinpath(@__DIR__, ".temp-input-data.zip");#tempname(); 

THRESHOLD::Float64 = 0.0;
function callback(total::Integer, now::Integer)
   global THRESHOLD;
   mbstep = 25.0;
   mbtotal = total / 10^6;
   mbnow = now / 10^6;
   if mbnow < THRESHOLD
      return;
   end
   THRESHOLD += mbstep;
   mbnow_r = round(mbnow, digits = 2);
   mbtotal_r = round(mbtotal, digits = 2);
   if (total == 0) 
       println("Downloading.. ($mbnow_r MB received)");
   else
       println("Downloading.. ($mbnow_r MB of total $mbtotal_r MB received)");
   end
end

# Download and extract.
filepath = Downloads.download(input_data_url, tempfile; progress = callback);
println("Finished downloading input data.");
println("Extracting contents of downloaded data.");
outfolder = joinpath(@__DIR__); 
InfoZIP.unzip(filepath, outfolder);
println("Downloaded files are in ", joinpath(outfolder, "input"));

# Remove tempfile.
rm(tempfile);
