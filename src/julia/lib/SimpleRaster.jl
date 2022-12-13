include("../../../config.jl");

import ArchGDAL
import GDAL
const AG = ArchGDAL;
using RCall
using DataFrames

"""
An object representing a bounding box.
"""
struct BoundingBox{T <: Real}
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    function BoundingBox(xmin::T, xmax::T, ymin::T, ymax::T) where {T <: Real}
        @assert xmin < xmax "`xmin` must be less than `xmax`";
        @assert ymin < ymax "`ymin` must be less than `ymax`";
        new{T}(xmin, xmax, ymin, ymax);
    end
end
xlength(bb::BoundingBox) = bb.xmax - bb.xmin;
ylength(bb::BoundingBox) = bb.ymax - bb.ymin;

function add_margin(bb::BoundingBox{T}, margin::T) where {T <: Real}
    BoundingBox(bb.xmin - margin, bb.xmax + margin,
                bb.ymin - margin, bb.ymax + margin);
end

"""
An information object for a SimpleRaster.
"""
struct RasterInformation
    bbox::BoundingBox{Float64} # Bounding box.
    nx::Int # Number of pixels in x direction.
    ny::Int # Number of pixels in y direction.
    pwh::Float64 # Pixel width and height (assumed the same).

    function RasterInformation(bbox::BoundingBox{Float64}, nx::Integer, ny::Integer, pwh::Float64)
        @assert nx == convert(Int, xlength(bbox) / pwh) "input `nx` does not match `bbox` and `pwh`";
        @assert ny == convert(Int, ylength(bbox) / pwh) "input `ny` does not match `bbox` and `pwh`";
        new(bbox, nx, ny, pwh);
    end
end

"""
Return the pixel indices where the point (x, y) falls.
y coordinate increases as row number increases.
x coordinate increases as column number increases.

Note values at exactly the lower boundary are deemed to be outside i.e
each pixel is a region (lowerx, upperx] × (lowery, uppery].
"""
function get_pixel_indices(info::RasterInformation, x::T, y::T) where {T <: Real}
    bbox = info.bbox;
    if x > bbox.xmax || x <= bbox.xmin
        col = -1;
    else
        col = ceil(Int, (x - bbox.xmin) / info.pwh);
    end
    if y > bbox.ymax || y <= bbox.ymin
        row = -1;
    else
        row = ceil(Int, (y - bbox.ymin) / info.pwh);
    end
    (row, col);
end

"""
Fetch information about a raster dataset.
Legend for the geotransform:
gt[1] = "x coordinate of the upper left corner of the upper left pixel"
gt[2] = "pixel width (x direction)"
gt[3] = "skew coefficient for x"
gt[4] = "y coordinate of the upper left corner of the upper left pixel"
gt[5] = "skew coefficient for y"
gt[6] = "pixel height (y direction)"

Note that the pixel widths can well be negative.
See:
https://gdal.org/doxygen/classGDALPamDataset.html (GetGeoTransform)
https://gis.stackexchange.com/questions/314654/gdal-getgeotransform-documentation-is-there-an-oversight-or-what-am-i-misund
"""
function RasterInformation(d::ArchGDAL.IDataset, bi::Int = 1)
    band = AG.getband(d, bi);
    gt = AG.getgeotransform(d);

    @assert gt[2] > 0.0 "SimpleRaster requires positive pixel width in geotransform";
    @assert gt[6] < 0.0 "SimpleRaster requires negative pixel height in geotransform.";
    @assert isapprox(gt[3], 0.0) "SimpleRaster does not support skew parameters, see documentation for RasterInformation.";
    @assert isapprox(gt[5], 0.0) "SimpleRaster does not support skew parameters, see documentation for RasterInformation.";

    xmin = gt[1]; ymax = gt[4];
    nx = convert(Int, AG.width(band));
    ny = convert(Int, AG.height(band));
    xmax = xmin + gt[2] * nx;
    ymin = ymax + gt[6] * ny;
    psx = (xmax - xmin) / nx;
    psy = (ymax - ymin) / ny;
    @assert psx == psy "RasterInformation only supports equal x and y pixel sizes.";
    pwh = psx;
    bbox = BoundingBox(xmin, xmax, ymin, ymax);
    RasterInformation(bbox, nx, ny, pwh);
end

"""
A simple raster object that follows the computer graphics convention that
the top left corner is the origin and x increases with increasing column
and y with increasing row.
"""
struct SimpleRaster{R <: Number}
    r::Matrix{R} # The raster data.
    mis::R # Value representing missing values in the raster.
    info::RasterInformation # Information about raster, bounding box.. etc.
end

# """
# Transform raster by substracting values from x and y limits.
# """
# function SimpleRaster(sr::SimpleRaster, x::Real, y::Real)
#     old_info = sr.info;
#     old_bbox = sr.info.bbox;
#     n_x = old_info.n_x; n_y = old_info.n_y;
#     ps_x = old_info.ps_x; ps_y = old_info.ps_y;
#     bbox = BoundingBox(old_bbox.xmin - x, old_bbox.xmax - x,
#                        old_bbox.ymin - y, old_bbox.ymax - y);
#     new_info = RasterInformation(bbox, n_x, n_y, ps_x, ps_y);
#     SimpleRaster(sr.r, sr.mis, new_info);
# end
#
# import Base.map
#
"""
Transform a SimpleRaster to a new SimpleRaster with the raster values and missing value
transformed with a lookup table.
"""
function transform(sr::SimpleRaster{R}, lookup::Dict{R, T}) where {R, T}
    new_mis = lookup[sr.mis];
    new_r = map(x -> lookup[x], sr.r);
    SimpleRaster(new_r, new_mis, deepcopy(sr.info));
end

"""
Transform a SimpleRaster to a new SimpleRaster with the raster values and missing value
transformed with a lookup table.
"""
function transform(sr::SimpleRaster{R}, f::Function) where R
    new_mis = f(sr.mis);
    new_r = map(f, sr.r);
    SimpleRaster(new_r, new_mis, deepcopy(sr.info));
end

# """
# Load an ArchGDAL band as a SimpleRaster object.
#
# Arguments:
# `d`: The ArchGDAL dataset.
# `bi`: The index of the band to load.
# """
# function SimpleRaster(d::ArchGDAL.Dataset, bi::Int = 1)
#     # Get raster information.
#     info = RasterInformation(d, bi);
#
#     # Load band from data.
#     r = AG.read(d, bi);
#     band = AG.getband(d, bi);
#     dtype = AG.getdatatype(band);
#     mis = convert(dtype, AG.getnodatavalue(band));
#
#     SimpleRaster{dtype, typeof(r), eltype(info)}(transpose(r)[info.n_y:-1:1, :], mis, info);
# end

"""
Load a subset specified by a bounding box from an ArchGDAL band as a SimpleRaster object.

Arguments:
`d`: The ArchGDAL dataset.
`bbox`: The bounding box covering the area to load.
`bi`: The index of the band to load.
"""
function SimpleRaster(d::ArchGDAL.IDataset, bi::Int, bbox::BoundingBox)
    # Get information about the full raster.
    # Here we also check that the orientation of the data is adequate
    # for SimpleRaster. Else an error will be thrown here.
    info = RasterInformation(d, bi);

    # Check bounding box given as a parameter, must be inside bounding box of
    # full raster.
    bbox_full = info.bbox; # Bounding box of the full raster.
    @assert bbox.xmin > bbox_full.xmin "xmin in `bbox` must be greater than the minimum in the bbox of the data";
    @assert bbox.ymin > bbox_full.ymin "ymin in `bbox` must be greater than the minimum in the bbox of the data";
    @assert bbox.ymax < bbox_full.ymax "ymax in `bbox` must be lesser than the maximum in the bbox of the data";
    @assert bbox.xmax < bbox_full.xmax "xmax in `bbox` must be lesser than the maximum in the bbox of the data";

    # Compute indices in the full raster which correspond to the bounding box.
    ymini, xmini = get_pixel_indices(info, bbox.xmin, bbox.ymin);
    ymaxi, xmaxi = get_pixel_indices(info, bbox.xmax, bbox.ymax);
    if ymini < 0 || xmini < 0 || ymaxi < 0 || xmaxi < 0
        throw(ArgumentError("something wrong. `get_pixel_indices` gave negative indices."))
    end
    # i_xmin = ceil(Int, (bbox.xmin - bbox_full.xmin) / info.pwh);
    # i_xmin = i_xmin == 0 ? i_xmin + 1 : i_xmin;
    # i_ymin = ceil(Int, (bbox.ymin - bbox_full.ymin) / info.pwh);
    # i_ymin = i_ymin == 0 ? i_ymin + 1 : i_ymin;
    # i_xmax = ceil(Int, (bbox.xmax - bbox_full.xmin) / info.pwh);
    # i_ymax = ceil(Int, (bbox.ymax - bbox_full.ymin) / info.pwh);

    # Compute the bbox that "snaps" to the grid specified by the raster pixels.
    # The bbox given as a parameter is likely not exactly at pixel boundaries.
    xmin = bbox_full.xmin + (xmini - 1) * info.pwh;
    ymin = bbox_full.ymin + (ymini - 1) * info.pwh;
    xmax = bbox_full.xmin + xmaxi * info.pwh;
    ymax = bbox_full.ymin + ymaxi * info.pwh;
    bbox_new = BoundingBox(xmin, xmax, ymin, ymax);

    # Build the RasterInformation object.
    nx = xmaxi - xmini + 1;
    ny = ymaxi - ymini + 1;
    info_new = RasterInformation(bbox_new, nx, ny, info.pwh);

    # Load raster data. Here the small y values in world coordinates are at
    # the bottom of the raster. (So a north up picture is correctly
    # orientated). Hence the flipped indexing to the pixels values.
    band = AG.getband(d, bi);
    dtype = AG.pixeltype(band);
    mis = convert(dtype, AG.getnodatavalue(band));
    r = AG.read(band, (info.ny - ymaxi + 1):(info.ny - ymini + 1), xmini:xmaxi);

    # Create SimpleRaster object.
    # The transpose is because for some reason AG.read transposes its output.
    # Here we also flip the y axis, so small y values in world coordinates
    # are at the top of the raster, as is the convention in computer graphics.
    SimpleRaster(transpose(r)[ny:-1:1, :], mis, info_new);
end

"""
Extract a value from a SimpleRaster{R} object.

Arguments:
`sr`: A SimpleRaster of type SimpleRaster{R}
`x`: The x coordinate of the point (of type T).
`y`: The y coordinate of the point (of type T).

Example:
extract(sr, 1000.823, 321.4)
"""
function extract(sr::SimpleRaster{R}, x::T, y::T) where {R <: Number, T <: Real}
    row, col = get_pixel_indices(sr.info, x, y);
    if row == -1 || col == -1
        return sr.mis;
    end
    sr.r[row, col];
end



"""
Function loads a full Corine Land Cover dataset as a SimpleRaster object.
"""
function load_corine_raster(year::Int)
    @assert year in (2012, 2018) "`year` must be 2012 or 2018."
    sr = ArchGDAL.registerdrivers() do
        ArchGDAL.read(CORINE_LAND_COVER_FILE(year)) do dataset
            SimpleRaster(dataset, 1);
        end
    end
    sr;
end

"""
Function loads a part of the Corine Land Cover dataset as a SimpleRaster object.
The part is specified by a BoundingBox object in the coordinates of the raster.
"""
function load_corine_raster(year::Int, bbox::BoundingBox)
    @assert year in (2012, 2018) "`year` must be 2012 or 2018."
    #sr = ArchGDAL.registerdrivers() do
        ArchGDAL.read(CORINE_LAND_COVER_FILE(year)) do dataset
            SimpleRaster(dataset, bbox, 1);
        end
    #end
    #sr;
end

"""
Function loads a Dict(Int, Float64) that maps the output from
the Corine rasters to the potential value given by the land type.
"""
function load_potential_lookup(corine::Int; outtype::DataType = Int, scaling::Function = log)
   @assert corine in (2012, 2018) "'corine' must be 2012 or 2018";
   filename = "corine" * string(corine) * "_prob_lookup.rds";
   potential_lookup = R"""readRDS(file.path($MISC_OUTPUT_PATH, $filename))""";
   n = parse.(outtype, string.(names(potential_lookup)));
   Dict(zip(n, scaling.(rcopy(potential_lookup))));
end

"""
Function loads a Dict{Int, Union{Missing, String}} that maps the Corine Land
Cover codes to the land type string given in the own land type classification.
"""
function load_landtype_lookup(corine::Int)
    @assert corine in (2012, 2018) "'corine' must be 2012 or 2018";
    filename = "corine" * string(corine) * "_landtype_lookup.rds";
    x = R"""readRDS(file.path($MISC_OUTPUT_PATH, $filename))""";
    n = parse.(Int, string.(names(x)));
    Dict(zip(n, rcopy(x)));
end
