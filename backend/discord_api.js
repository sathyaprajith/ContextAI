/**
 * ContextAI - Discord Bot Backend
 * Team Gear5
 */

const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const { Client, GatewayIntentBits } = require("discord.js");
const { GoogleGenerativeAI } = require("@google/generative-ai");

require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));

// RAG Server Integration
const INGEST_SERVER_URL = "http://localhost:3000/api/v1/ingest";
const DISCORD_MAPPING_FILE = path.join(__dirname, "discord_mapping.json");

// Load Discord-to-Project mapping
let discordMapping = {};
if (fs.existsSync(DISCORD_MAPPING_FILE)) {
  try {
    discordMapping = JSON.parse(fs.readFileSync(DISCORD_MAPPING_FILE, "utf8"));
    console.log(
      `📋 Loaded ${Object.keys(discordMapping).length} Discord channel mappings`
    );
  } catch (err) {
    console.warn("⚠️ Failed to load discord_mapping.json:", err.message);
  }
} else {
  console.log("📋 No discord_mapping.json found - RAG features disabled");
}

// Storage directory for Discord summaries
const SUMMARIES_DIR = path.join(__dirname, "contextai_discord_summaries");
if (!fs.existsSync(SUMMARIES_DIR)) {
  fs.mkdirSync(SUMMARIES_DIR, { recursive: true });
}

// Initialize Gemini AI with multi-key support
const geminiAPIKeys = [];
const geminiModels = [];
let currentModelIndex = 0;

if (process.env.GEMINI_API_KEY) geminiAPIKeys.push(process.env.GEMINI_API_KEY);
if (process.env.GEMINI_API_KEY_2)
  geminiAPIKeys.push(process.env.GEMINI_API_KEY_2);
if (process.env.GEMINI_API_KEY_3)
  geminiAPIKeys.push(process.env.GEMINI_API_KEY_3);

if (geminiAPIKeys.length === 0) {
  console.warn("⚠️ No GEMINI_API_KEY found - AI analysis disabled");
} else {
  geminiAPIKeys.forEach((key, index) => {
    try {
      const genAI = new GoogleGenerativeAI(key);
      const model = genAI.getGenerativeModel({
        model: "gemini-2.5-flash-lite",
      });
      geminiModels.push(model);
      console.log(`🤖 Gemini AI #${index + 1} initialized`);
    } catch (err) {
      console.error(
        `❌ Failed to initialize API key #${index + 1}:`,
        err.message
      );
    }
  });
  console.log(
    `✅ Total active API keys: ${geminiModels.length}/${geminiAPIKeys.length}`
  );
}

function getGeminiModel() {
  if (geminiModels.length === 0) return null;
  currentModelIndex = (currentModelIndex + 1) % geminiModels.length;
  console.log(
    `   🔄 Using Gemini API #${currentModelIndex + 1} of ${geminiModels.length}`
  );
  return geminiModels[currentModelIndex];
}

// Retry helper for API calls with exponential backoff
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 2000) {
  let lastError = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const isRetryable =
        error.message.includes("503") ||
        error.message.includes("500") ||
        error.message.includes("overloaded") ||
        error.message.includes("429") ||
        error.message.includes("RESOURCE_EXHAUSTED");

      if (attempt === maxRetries || !isRetryable) {
        throw lastError;
      }

      const delay = baseDelay * Math.pow(2, attempt - 1);
      console.log(
        `⚠️ API error (attempt ${attempt}/${maxRetries}), retrying in ${
          delay / 1000
        }s...`
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

// Initialize Discord Bot
const DISCORD_TOKEN = process.env.DISCORD_BOT_TOKEN;

if (!DISCORD_TOKEN) {
  console.error("❌ DISCORD_BOT_TOKEN not found in .env file");
  console.log(
    "⚠️  Please add DISCORD_BOT_TOKEN=your_token_here to backend/.env"
  );
  process.exit(1);
}

const discordClient = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

let isDiscordReady = false;
let guildsCache = [];
let channelsCache = {};

discordClient.once("ready", () => {
  console.log(`✅ Discord Bot logged in as ${discordClient.user.tag}`);
  isDiscordReady = true;

  // Cache guilds and channels
  guildsCache = Array.from(discordClient.guilds.cache.values()).map(
    (guild) => ({
      id: guild.id,
      name: guild.name,
      icon: guild.iconURL(),
      memberCount: guild.memberCount,
    })
  );

  guildsCache.forEach((guild) => {
    const guildObj = discordClient.guilds.cache.get(guild.id);
    channelsCache[guild.id] = Array.from(guildObj.channels.cache.values())
      .filter((ch) => ch.type === 0) // Only GuildText channels (ChannelType.GuildText = 0)
      .map((ch) => ({
        id: ch.id,
        name: ch.name,
        type: ch.type,
      }));
  });

  console.log(`📡 Connected to ${guildsCache.length} servers`);
});

// Auto-ingest Discord messages to RAG
discordClient.on("messageCreate", async (message) => {
  try {
    // Ignore bot messages
    if (message.author.bot) return;

    // Only process text channels
    if (message.channel.type !== 0) return;

    const channelKey = `${message.guildId}_${message.channelId}`;
    const mapping = discordMapping[channelKey];

    if (mapping && message.content) {
      console.log(
        `[DISCORD-INGEST] Channel ${channelKey} IS mapped. Sending to RAG server...`
      );

      const payload = {
        project_id: mapping.project_id,
        team_id: mapping.team_id,
        source: "discord",
        text: message.content,
      };

      axios
        .post(INGEST_SERVER_URL, payload)
        .then(() => {
          console.log(
            `[DISCORD-INGEST] SUCCESS: Sent message from ${message.channel.name} to RAG server.`
          );
        })
        .catch((err) => {
          console.error(
            "[DISCORD-INGEST] FAILED to send to RAG server:",
            err.message
          );
        });
    } else if (message.content) {
      console.log(
        `[DISCORD-INGEST] Channel ${channelKey} is NOT in discord_mapping.json. Ignoring.`
      );
    }
  } catch (err) {
    console.error(
      "[DISCORD-INGEST] Error processing message for ingestion:",
      err
    );
  }
});

discordClient.on("error", (error) => {
  console.error("❌ Discord client error:", error);
});

// Login to Discord
discordClient.login(DISCORD_TOKEN).catch((err) => {
  console.error("❌ Failed to login to Discord:", err.message);
});

// API Endpoints

// Get Discord bot status
app.get("/api/status", (req, res) => {
  res.json({
    ready: isDiscordReady,
    user: isDiscordReady
      ? {
          id: discordClient.user.id,
          username: discordClient.user.username,
          tag: discordClient.user.tag,
          avatar: discordClient.user.displayAvatarURL(),
        }
      : null,
    guilds: guildsCache.length,
  });
});

// Get all servers (guilds)
app.get("/api/guilds", (req, res) => {
  if (!isDiscordReady) {
    return res.status(503).json({ error: "Discord bot not ready" });
  }

  res.json({ guilds: guildsCache });
});

// Get channels for a specific guild
app.get("/api/guilds/:guildId/channels", (req, res) => {
  const { guildId } = req.params;

  if (!isDiscordReady) {
    return res.status(503).json({ error: "Discord bot not ready" });
  }

  const channels = channelsCache[guildId] || [];
  res.json({ channels });
});

// Analyze channel messages
app.post("/api/analyze", async (req, res) => {
  try {
    const {
      guildId,
      channelId,
      analysisDepth = "moderate",
      messageLimit = 100,
    } = req.body;

    if (!guildId || !channelId) {
      return res.status(400).json({ error: "guildId and channelId required" });
    }

    if (!isDiscordReady) {
      return res.status(503).json({ error: "Discord bot not ready" });
    }

    const guild = discordClient.guilds.cache.get(guildId);
    if (!guild) {
      return res.status(404).json({ error: "Guild not found" });
    }

    const channel = guild.channels.cache.get(channelId);
    if (!channel) {
      return res.status(404).json({ error: "Channel not found" });
    }

    console.log(`📊 Analyzing #${channel.name} in ${guild.name}...`);

    // Fetch messages
    const messages = await channel.messages.fetch({
      limit: Math.min(messageLimit, 100),
    });
    const messageArray = Array.from(messages.values()).reverse();

    if (messageArray.length === 0) {
      return res.status(400).json({ error: "No messages found in channel" });
    }

    // Format messages for analysis
    const formattedMessages = messageArray
      .map((msg) => {
        const timestamp = msg.createdAt.toLocaleString();
        const author = msg.author.username;
        const content = msg.content || "[Attachment/Embed]";
        return `[${timestamp}] ${author}: ${content}`;
      })
      .join("\n");

    // Get AI model
    const model = getGeminiModel();
    if (!model) {
      return res.status(503).json({ error: "AI not configured" });
    }

    const isDeepAnalysis = analysisDepth === "deep";

    let prompt;
    if (isDeepAnalysis) {
      prompt = `Analyze this Discord channel conversation in DEEP RESEARCH mode:

Server: ${guild.name}
Channel: #${channel.name}
Messages analyzed: ${messageArray.length}

Conversation:
${formattedMessages}

Provide an IN-DEPTH analysis with:
**📊 Deep Context Analysis**
* Overall purpose and background
* Community dynamics

**📋 Detailed Topics & Themes**
* Main topics with context
* Technical discussions
* Connections between topics

**✅ Decisions & Reasoning**
* Decisions made with reasoning
* Impact and implications

**📌 Action Items & Dependencies**
* Tasks with owners and deadlines

**👥 Active Participants**
* Key contributors and roles

**💡 Critical Insights**
* Important moments with deeper meaning

**🔍 Patterns & Predictions**
* Recurring themes
* Community health indicators

Keep well-structured with clear bullet points.`;
    } else {
      prompt = `Analyze this Discord channel conversation and provide a concise summary:

Server: ${guild.name}
Channel: #${channel.name}
Messages analyzed: ${messageArray.length}

Conversation:
${formattedMessages}

Provide a summary with:
**📊 Overview**
* Main purpose and context

**📋 Key Topics**
* Important topics discussed

**✅ Decisions & Action Items**
* Decisions made and next steps

**👥 Active Participants**
* Key contributors

**💡 Highlights**
* Notable quotes or moments

Keep it concise and well-structured.`;
    }

    console.log(`🤖 Analyzing with Gemini AI (${analysisDepth} mode)...`);

    const result = await retryWithBackoff(
      async () => await model.generateContent(prompt),
      3,
      2000
    );

    const response = await result.response;
    const summary = response.text();

    // Save summary
    const summaryId = `discord_${guildId}_${channelId}_${Date.now()}`;
    const summaryData = {
      id: summaryId,
      guildId,
      guildName: guild.name,
      channelId,
      channelName: channel.name,
      messageCount: messageArray.length,
      summary,
      analysisDepth,
      messageLimit,
      analyzedAt: new Date().toISOString(),
    };

    fs.writeFileSync(
      path.join(SUMMARIES_DIR, `${summaryId}.json`),
      JSON.stringify(summaryData, null, 2)
    );

    // Save to dashboard summaries
    const dashboardSummary = {
      chatId: summaryId,
      chatName: `#${channel.name} (${guild.name})`,
      summary: summary,
      stats: {
        messageCount: messageArray.length,
        server: guild.name,
        channel: channel.name,
        date: new Date().toISOString(),
      },
      source: "discord",
      timestamp: new Date().toISOString(),
    };

    const summariesPath = path.join(__dirname, "summaries.json");
    let allSummaries = [];
    try {
      if (fs.existsSync(summariesPath)) {
        const data = fs.readFileSync(summariesPath, "utf8");
        allSummaries = JSON.parse(data);
      }
    } catch (err) {
      allSummaries = [];
    }

    allSummaries = allSummaries.filter((s) => s.chatId !== summaryId);
    allSummaries.unshift(dashboardSummary);
    fs.writeFileSync(summariesPath, JSON.stringify(allSummaries, null, 2));

    console.log(`✅ Analysis complete and saved to dashboard`);
    res.json({ success: true, summary, summaryId });
  } catch (err) {
    console.error("Analysis error:", err);
    res.status(500).json({ error: "Analysis failed", message: err.message });
  }
});

// Q&A on channel - NOW POWERED BY RAG SERVER
app.post("/api/qa", async (req, res) => {
  try {
    const { summaryId, question, guildId, channelId } = req.body;

    if (!question) {
      return res.status(400).json({ error: "question required" });
    }

    console.log(`[DISCORD-RAG] New Q&A Request`);
    console.log(`[DISCORD-RAG] Question: ${question}`);

    // Check if we have RAG mapping for this channel
    const channelKey = `${guildId}_${channelId}`;
    const mapping = discordMapping[channelKey];

    if (mapping) {
      // Use RAG server for mapped channels
      console.log(
        `[DISCORD-RAG] Channel ${channelKey} is mapped. Using RAG server...`
      );

      const ragPayload = {
        project_id: mapping.project_id,
        team_id: mapping.team_id,
        question: question,
      };

      try {
        const ragResponse = await axios.post(
          "http://localhost:3000/api/v1/ask",
          ragPayload
        );

        console.log("[DISCORD-RAG] Answer generated from RAG server");
        return res.json({
          success: true,
          question: question,
          answer: ragResponse.data.answer,
        });
      } catch (err) {
        console.error("❌ DISCORD-RAG Error:", err.message);
        // Fall back to Gemini if RAG fails
      }
    }

    // Fallback: Use Gemini with summary (for unmapped channels or if RAG fails)
    if (!summaryId) {
      return res
        .status(400)
        .json({ error: "summaryId required for non-RAG channels" });
    }

    const filePath = path.join(SUMMARIES_DIR, `${summaryId}.json`);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Summary not found" });
    }

    const summaryData = JSON.parse(fs.readFileSync(filePath, "utf8"));
    const model = getGeminiModel();

    if (!model) {
      return res.status(503).json({ error: "AI not configured" });
    }

    console.log("[DISCORD-RAG] Using Gemini fallback with summary...");
    const prompt = `Discord Channel: #${summaryData.channelName} in ${summaryData.guildName}

Summary:
${summaryData.summary}

Question: ${question}

Provide a concise, direct answer based on the summary. If the information isn't in the summary, say so.`;

    const result = await retryWithBackoff(
      async () => await model.generateContent(prompt),
      3,
      2000
    );

    const response = await result.response;
    const answer = response.text();

    res.json({ success: true, question, answer });
  } catch (err) {
    console.error("Q&A error:", err);
    res.status(500).json({ error: "Q&A failed", message: err.message });
  }
});

// Delete summary
app.delete("/api/summaries/:summaryId", (req, res) => {
  try {
    const { summaryId } = req.params;
    const filePath = path.join(SUMMARIES_DIR, `${summaryId}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Summary not found" });
    }

    fs.unlinkSync(filePath);
    console.log(`🗑️ Deleted summary: ${summaryId}`);
    res.json({ success: true });
  } catch (err) {
    console.error("Delete error:", err);
    res.status(500).json({ error: "Delete failed" });
  }
});

// Get all summaries
app.get("/api/summaries", (req, res) => {
  try {
    const files = fs
      .readdirSync(SUMMARIES_DIR)
      .filter((f) => f.endsWith(".json"));
    const summaries = files
      .map((file) => {
        const data = JSON.parse(
          fs.readFileSync(path.join(SUMMARIES_DIR, file), "utf8")
        );
        return {
          id: data.id,
          guildName: data.guildName,
          channelName: data.channelName,
          messageCount: data.messageCount,
          analyzedAt: data.analyzedAt,
          summary: data.summary,
        };
      })
      .sort((a, b) => new Date(b.analyzedAt) - new Date(a.analyzedAt));

    res.json({ summaries });
  } catch (err) {
    console.error("Error loading summaries:", err);
    res.status(500).json({ error: "Failed to load summaries" });
  }
});

const PORT = 8004;

app.listen(PORT, () => {
  console.log("🚀 ContextAI Discord Analyzer");
  console.log("===================================");
  console.log(`📡 API Server: http://localhost:${PORT}`);
  console.log("===================================");
  console.log("Team: Gear5\n");
});
