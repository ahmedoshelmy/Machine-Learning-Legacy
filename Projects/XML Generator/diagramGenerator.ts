import {  LLMInstance } from "./invokeLLM";
import { WriteXML } from "./utils";

export async function generateDiagram(userInput: string) {
  try {
    const llmInstance = new LLMInstance();

    const response = await llmInstance.callLLM(userInput);
    console.log("LLM Response:", response);
    WriteXML(response);
    return response;
  } catch (error) {
    console.error("Error:", error);
  }
}


// Giving some examples for illustration
(async () => {
  const llmInstance = new LLMInstance();

  const prompt1 = "Design a UML diagram for a library management system.";
  const xml1 = await llmInstance.callLLM(prompt1);
  console.log("Generated XML 1:", xml1);
  WriteXML(xml1,'xml1.drawio')

  const prompt2 = "Now, add user entity to the system";
  const xml2 = await llmInstance.callLLM(prompt2);
  console.log("Generated XML 2:", xml2);
  WriteXML(xml2,'xml2.drawio')

  const prompt3 = "Now, remove user entity to the system";
  const xml3 = await llmInstance.callLLM(prompt3);
  console.log("Generated XML 3:", xml3);
  WriteXML(xml3,'xml3.drawio')

  const prompt4 = "Add the sketch attribute to make the design look like a sketch";
  const xml4 = await llmInstance.callLLM(prompt4);
  console.log("Generated XML 4:", xml4);
  WriteXML(xml4,'xml4.drawio')

  // If you want to regenerate the last XML code
  const regeneratedXML = await llmInstance.regenerateXML();
  console.log("Regenerated XML:", regeneratedXML);
  WriteXML(regeneratedXML,'xml4.drawio')

})();
