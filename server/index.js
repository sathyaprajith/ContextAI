// Import the dotenv library to read .env files
require("dotenv").config();

// Import express
const express = require("express");

// Import the "Pool" class from the pg library
const { Pool } = require("pg");

// Import the Google Generative AI library
const { GoogleGenerativeAI } = require("@google/generative-ai");

// --- SETUP ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 1. The "Analyst" model (for extracting JSON)
const extractionModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

// 2. The "Embedding" model (for creating vectors)
const embeddingModel = genAI.getGenerativeModel({
  model: "text-embedding-004",
});

// Create a new pool instance for the database
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

const app = express();
const PORT = 3000;
app.use(express.json()); // Middleware to parse JSON

// --- API ENDPOINTS (PART 1: INGESTION) ---

app.post("/api/v1/ingest", async (req, res) => {
  try {
    const { project_id, team_id, source, text } = req.body;
    if (!project_id || !team_id || !text) {
      return res
        .status(400)
        .json({ error: "Missing project_id, team_id, or text" });
    }
    console.log(`Received data for project: ${project_id}`);

    // STEP 1: AI Analysis
    console.log("Sending text to AI for analysis...");
    const extractionPrompt = `
      Analyze the following text. Extract any tasks, decisions, or important facts 
      as a JSON array. 
      - Tasks should have a "type": "TASK" and "content".
      - Decisions should have a "type": "DECISION" and "content".
      - Facts should have a "type": "FACT" and "content".
      - If no specific event is found, return an empty array [].
      Text to analyze: "${text}"
    `;
    const result = await extractionModel.generateContent(extractionPrompt);
    const response = result.response;
    let aiResponseText = response.text();

    // Using .replaceAll() is safer than regex here
    aiResponseText = aiResponseText
      .replaceAll("```json", "")
      .replaceAll("```", "")
      .trim();

    const events = JSON.parse(aiResponseText);

    if (events.length === 0) {
      console.log("AI found no events. Nothing to save.");
      return res.status(200).json({ message: "No events found in text." });
    }
    console.log(`AI found ${events.length} event(s). Processing...`);

    // STEPS 2 & 3: Embed & Save
    for (const event of events) {
      const contentToEmbed = event.content;
      console.log(`Embedding content: "${contentToEmbed}"`);
      const embeddingResult = await embeddingModel.embedContent(contentToEmbed);
      const embedding = embeddingResult.embedding.values;
      const embeddingString = `[${embedding.join(",")}]`;

      console.log("Saving event to database...");
      const insertQuery = `
        INSERT INTO events (project_id, team_id, source, event_data, embedding)
        VALUES ($1, $2, $3, $4, $5)
      `;
      await pool.query(insertQuery, [
        project_id,
        team_id,
        source,
        event,
        embeddingString,
      ]);
      console.log("Event saved successfully!");
    }
    res.status(200).json({
      message: `Successfully processed and saved ${events.length} event(s).`,
      savedEvents: events,
    });
  } catch (error) {
    console.error("Error in /ingest endpoint:", error);
    res.status(500).json({ error: "Failed to process data" });
  }
});

// --- UPDATED ENDPOINT (PART 2: RETRIEVAL) ---

app.post("/api/v1/ask", async (req, res) => {
  try {
    // 1. Get user's question and "folder" filters
    const { project_id, team_id, question } = req.body;

    if (!project_id || !question) {
      return res.status(400).json({ error: "Missing project_id or question" });
    }
    console.log(`New question for project ${project_id}: "${question}"`);

    // 2. Embed the user's question
    console.log("Embedding the question...");
    const embeddingResult = await embeddingModel.embedContent(question);
    const questionEmbedding = `[${embeddingResult.embedding.values.join(",")}]`;

    // 3. Search the database for relevant facts
    console.log("Searching database for relevant context...");
    let queryParams = [project_id, questionEmbedding];
    let similarityQuery = `
      SELECT 
        event_data,
        timestamp,  -- <-- ADD THIS LINE
        1 - (embedding <-> $2) AS similarity
      FROM events
      WHERE project_id = $1
    `;

    // If a team_id is provided, add it to the filter
    if (team_id) {
      similarityQuery += ` AND team_id = $3`;
      queryParams.push(team_id);
    }

    // Order by similarity and get the top 5
    similarityQuery += ` ORDER BY similarity DESC LIMIT 5`;

    const searchResults = await pool.query(similarityQuery, queryParams);

    searchResults.rows.sort(
      (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
    );

    if (searchResults.rows.length === 0) {
      console.log("No context found.");
      return res.status(200).json({
        answer:
          "I'm sorry, I couldn't find any information relevant to your question.",
      });
    }

    // 4. Build a new prompt for Gemini
    console.log("Building prompt with context...");
    let context = "Context:\n";
    for (const row of searchResults.rows) {
      // Add each event's JSON data to the context
      context += `- ${JSON.stringify(row.event_data)}\n`;
    }

    const finalPrompt = `
      ${context}
      
      User's Question: "${question}"

      Based *only* on the context provided above, answer the user's question. 
      If the context doesn't contain the answer, say you couldn't find the information.
    `;

    // 5. Call Gemini to get the final answer
    console.log("Generating final answer...");
    const result = await extractionModel.generateContent(finalPrompt);
    const answer = result.response.text();

    console.log("Answer generated:", answer);

    // --- THIS IS THE CORRECTED LINE ---
    res.status(200).json({ answer: answer });
  } catch (error) {
    console.error("Error in /ask endpoint:", error);
    res.status(500).json({ answer: "Failed to answer question" });
  }
});

// --- (Existing routes for testing) ---
app.get("/testdb", async (req, res) => {
  try {
    const result = await pool.query("SELECT NOW()");
    res.json({
      message: "Database connection successful!",
      time: result.rows[0].now,
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("Database connection failed!");
  }
});

app.get("/", (req, res) => {
  res.send("Hello, World! Your server is running!");
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
