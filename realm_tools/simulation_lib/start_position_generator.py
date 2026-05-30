"""
start_position_generator.py

Generates a uniform grid of valid robot start positions for a maze
defined in a REALM environment XML file.

Algorithm
---------
1. Parse all wall segments from the XML.
2. Use Shapely polygonize to find the closed outer boundary polygon formed
   by whichever wall segments connect end-to-end into a ring.
3. Erode the boundary inward by `wall_clearance` so the robot body
   stays away from the perimeter.
4. Buffer every wall segment outward by `wall_clearance` and subtract
   those regions — keeps the robot clear of interior obstacle walls too.
5. Lay a uniform grid over the remaining safe zone and keep every point
   that falls inside it.
6. Save to CSV  (x, y, theta).

Usage — from Python
-------------------
    from realm_tools.simulation_lib.start_position_generator import generate_grid

    df = generate_grid(
        'simulation/worlds/environments/vpce/LMO8.xml',
        spacing=0.1,
        wall_clearance=0.5,
        output='simulation/worlds/environments/vpce/LMO8_train_points.csv'
    )

Usage — from the command line
------------------------------
    python -m realm_tools.simulation_lib.start_position_generator \\
        simulation/worlds/environments/vpce/LMO8.xml \\
        --spacing 0.1 --clearance 0.5 --theta 0.0 \\
        --output simulation/worlds/environments/vpce/LMO8_train_points.csv
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union

from realm_tools.simulation_lib.environment_parser import parse_all_walls

# Robot physical constants (from HamBot spec)
ROBOT_RADIUS   = 0.3086   # metres
MIN_LIDAR_DIST = 0.50     # metres — minimum safe lidar clearance for actions


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wall_lines(walls):
    """Return a list of Shapely LineStrings for all wall segments."""
    return [LineString([(w['x1'], w['y1']), (w['x2'], w['y2'])]) for w in walls]


def _outer_boundary(walls):
    """
    Build the outer boundary polygon from the maze walls.

    Prefers walls tagged type="boundary".  Falls back to all walls if none
    are tagged (older XML format).  Uses the convex hull of the selected
    wall endpoints — this is exact for convex environments (all current VPCE
    mazes are convex octagons) and requires no exact endpoint connectivity.
    """
    boundary_walls = [w for w in walls if w['wall_type'] == 'boundary']
    source         = boundary_walls if boundary_walls else walls

    endpoints = (
        [(w['x1'], w['y1']) for w in source] +
        [(w['x2'], w['y2']) for w in source]
    )
    return MultiPoint(endpoints).convex_hull


def _is_valid_point(x, y, boundary, wall_lines, wall_clearance):
    """
    Return True if (x, y) is a valid robot start position:
      - inside the outer boundary polygon, and
      - at least wall_clearance metres from every wall segment.

    Using per-point distance checks avoids the Shapely buffer/difference
    approach, which merges clearance zones from opposing walls and incorrectly
    eliminates narrow-but-navigable corridors.
    """
    p = Point(x, y)
    if not boundary.contains(p):
        return False
    return all(p.distance(line) >= wall_clearance for line in wall_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_grid(xml_path, spacing=0.1, wall_clearance=MIN_LIDAR_DIST,
                  theta=0.0, output=None):
    """
    Generate a uniform grid of valid start positions for a REALM maze.

    Parameters
    ----------
    xml_path : str
        Path to the maze XML file.
    spacing : float
        Distance between grid points in metres.  Smaller = denser coverage.
        Default 0.1 m.
    wall_clearance : float
        Minimum distance from any wall surface to a grid point in metres.
        Should be at least the robot radius (0.31 m).
        Default 0.5 m (matches the minimum lidar action distance).
    theta : float
        Heading assigned to every generated position, in radians.
        Default 0.0.
    output : str or None
        If given, save the result to this CSV path (directories are created
        automatically).  Either way the DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        Columns: x, y, theta.
    """
    root  = ET.parse(xml_path).getroot()
    walls = parse_all_walls(root)
    lines = _wall_lines(walls)

    boundary = _outer_boundary(walls)

    # Restrict the search grid to the boundary bounding box, shrunk by
    # wall_clearance on each side (no valid point can exist outside this).
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(minx + wall_clearance, maxx - wall_clearance + spacing, spacing)
    ys = np.arange(miny + wall_clearance, maxy - wall_clearance + spacing, spacing)

    points = [
        (round(float(x), 4), round(float(y), 4), theta)
        for x in xs
        for y in ys
        if _is_valid_point(x, y, boundary, lines, wall_clearance)
    ]

    if not points:
        raise ValueError(
            f"No valid positions found with wall_clearance={wall_clearance} m. "
            "Try reducing wall_clearance."
        )

    df = pd.DataFrame(points, columns=['x', 'y', 'theta'])

    if output:
        dir_name = os.path.dirname(output)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Saved {len(df)} positions  →  {output}")

    return df


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Generate a CSV of valid robot start positions for a REALM maze XML.\n\n"
            "If --output is omitted the file is written alongside the XML as "
            "<maze>_train_points.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('xml',
                   help='Path to the maze XML file.')
    p.add_argument('--spacing', type=float, default=0.1,
                   help='Grid spacing in metres (default: 0.1).')
    p.add_argument('--clearance', type=float, default=MIN_LIDAR_DIST,
                   help=f'Min distance from any wall in metres '
                        f'(default: {MIN_LIDAR_DIST}).')
    p.add_argument('--theta', type=float, default=0.0,
                   help='Robot heading for all positions in radians (default: 0.0).')
    p.add_argument('--output', '-o', default=None,
                   help='Output CSV path.  Defaults to <maze>_train_points.csv '
                        'alongside the input XML.')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    output = args.output
    if output is None:
        output = os.path.splitext(args.xml)[0] + '_train_points.csv'

    generate_grid(
        xml_path=args.xml,
        spacing=args.spacing,
        wall_clearance=args.clearance,
        theta=args.theta,
        output=output,
    )
