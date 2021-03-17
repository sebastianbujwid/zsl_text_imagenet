#!/usr/bin/env bash

wikipedia_xml="$1"

# Info: Use full path to keep the full path in meta_json_file

if [[ ! -f "${wikipedia_xml}" ]]; then
  echo "Cannot find ${wikipedia_xml}"
  exit 1
fi

data_dir=$(dirname "${wikipedia_xml}")
xml_name=$(basename "${wikipedia_xml}" ".xml")
log_file="${data_dir}/wikiextractor_${xml_name}.log"
meta_json_file="${data_dir}/meta_${xml_name}.json"
output_dir="${data_dir}/extracted_${xml_name}"

if [[ -f "${meta_json_file}" ]]; then
    echo "File ${meta_json_file} already exists!"
    exit 2
fi

if [[ -d "${output_dir}" ]]; then
  echo "Directory ${output_dir} already exists!"
  exit 2
fi
mkdir "${output_dir}"

echo "Extracting text from: ${wikipedia_xml}"
python -m wikiextractor.WikiExtractor --json --sections --lists -o "${output_dir}" --log_file "${log_file}" "${wikipedia_xml}"
num_articles=$(tail "${log_file}" -n 1 | awk -F";" '{print $2}' | awk -F": " '{print $2}')

echo "Extracting article titles, IDs and file paths..."

awk -F', "text"' '{print $1",", "\"file\": \"" FILENAME"\"}"}' "${output_dir}"/**/* > "${meta_json_file}"

if [[ $(cat "${meta_json_file}" | wc -l) -eq "${num_articles}" ]]; then
    echo "The number of articles matches: ${num_articles}!"
else
    echo "Could not match the number of articles!"
    exit 3
fi

echo "Saving meta wiki into a pickle"
pickle_meta_script=$(dirname $0)"/pickle_meta_wiki_file.py"
python "${pickle_meta_script}" "--meta_wiki_json" "${meta_json_file}"

echo "Extracting pages metadata..."
extract_pages_metadata_script=$(dirname $0)"/extract_pages_metadata.py"
python "${extract_pages_metadata_script}" "--wikipedia_xml" "${wikipedia_xml}"
