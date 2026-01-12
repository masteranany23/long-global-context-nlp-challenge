import pathway as pw

# Test reading novels
novels_table = pw.io.fs.read(
    path="./data/novels",
    format="plaintext_by_file",
    with_metadata=True,
    mode="static"
)

# Print schema
print("Novels table created")

# Write to debug file
pw.io.csv.write(novels_table, "debug_novels.csv")

# Run
pw.run()

print("Done")
