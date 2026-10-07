using Printf

"""
    pdbline(record, serial, atomname, resname, chain, resnum, x, y, z, element)

Format one fixed-width PDB ATOM/HETATM record. Test-only helper — production
parsing always goes through real structure files via BioStructures.jl.
"""
function pdbline(record, serial, atomname, resname, chain, resnum, x, y, z, element)
    name_field = length(atomname) >= 4 ? atomname : @sprintf(" %-3s", atomname)
    @sprintf(
        "%-6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n",
        record, serial, name_field, resname, chain, resnum, x, y, z, 1.0, 0.0, element,
    )
end

const SAMPLE_PDB = join(
    [
        pdbline("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        pdbline("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
        pdbline("ATOM", 3, "C", "ALA", "A", 1, 2.0, 0.0, 0.0, "C"),
        pdbline("ATOM", 4, "O", "ALA", "A", 1, 3.0, 0.0, 0.0, "O"),
        pdbline("ATOM", 5, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0, "C"),
        pdbline("ATOM", 6, "N", "GLY", "A", 2, 4.0, 0.0, 0.0, "N"),
        pdbline("ATOM", 7, "CA", "GLY", "A", 2, 5.0, 0.0, 0.0, "C"),
        pdbline("ATOM", 8, "C", "GLY", "A", 2, 6.0, 0.0, 0.0, "C"),
        pdbline("ATOM", 9, "O", "GLY", "A", 2, 7.0, 0.0, 0.0, "O"),
        pdbline("HETATM", 10, "ZN", "ZN", "B", 1, 10.0, 10.0, 10.0, "ZN"),
        pdbline("HETATM", 11, "FE", "HEM", "C", 1, 12.0, 12.0, 12.0, "FE"),
        pdbline("HETATM", 12, "N1", "HEM", "C", 1, 13.0, 12.0, 12.0, "N"),
        pdbline("HETATM", 13, "O", "HOH", "D", 1, 20.0, 20.0, 20.0, "O"),
    ],
) * "END\n"
