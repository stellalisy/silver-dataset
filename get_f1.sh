output_file=${1}

echo `tail -n 1 ${output_file} | cut -d',' -f4`