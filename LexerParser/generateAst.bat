@echo off
echo Generating AST images...

for %%f in (ast*.dot) do (
    echo Converting %%f...
    dot -Tpng "%%f" -o "%%~nf.png"
)

echo Done! All AST PNGs generated.
pause
