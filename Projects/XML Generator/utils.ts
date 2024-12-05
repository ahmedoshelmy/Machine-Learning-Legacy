import fs from 'fs';
import path from 'path';

/**
 * Writes the provided XML content to a `.drawio` file.
 * @param {string} xmlContent - The XML content to be written to the file.
 * @param {string} [filename='diagram.drawio'] - The name of the file to create. Defaults to 'diagram.drawio'.
 */
export function WriteXML(xmlContent:string, filename = 'diagram.drawio') {
    // Ensure the file has the correct extension
    const filePath = path.extname(filename) === '.drawio' ? filename : `${filename}.drawio`;
    try {
        // Write the XML content to the file
        fs.writeFileSync(filePath, xmlContent, { encoding: 'utf8' });
        console.log(`File written successfully to ${filePath}`);
    } catch (error) {
        console.error('Error writing the file:', error);
    }
}
