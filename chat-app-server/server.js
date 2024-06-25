const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const socket = require("./socketManager");
const _ = require("lodash");

require('dotenv').config();

const authRoutes = require("./routes/auth_users");
const conversationRoutes = require("./routes/conversations");
const messageRoutes = require("./routes/messages");
const analysisRoutes = require("./routes/data_analysis");

const app = express();

app.use(cors());
app.use(express.json());

app.use("/api/auth", authRoutes);
app.use("/api/conversations", conversationRoutes);
app.use("/api/messages", messageRoutes);
app.use("/api/sentimentanalysis", analysisRoutes);

mongoose
  .connect(process.env.MONGO_URL)
  .then(() => {
    console.log("DB Connection Successfull");
  })
  .catch((err) => {
    console.log(err.message);
  });

const server = app.listen(process.env.SERVER_PORT, () => {
  console.log(`Server is running on port ${process.env.SERVER_PORT}`);
});

socket.initSocket(server);
