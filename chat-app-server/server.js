const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const socket = require("socket.io");

const authRoutes = require("./routes/auth_users");
const messageRoutes = require("./routes/messages");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

const url = "mongodb+srv://admin:admin@cluster.usyyx.mongodb.net/ChatApp?retryWrites=true&w=majority"
mongoose
  .connect(url)
  .then(() => {
    console.log("DB Connection Successfull");
  })
  .catch((err) => {
    console.log(err.message);
  });

app.use("/api/auth", authRoutes);
app.use("/api/messages", messageRoutes);

app.get('/', (req, res) => {
  res.send('Welcome to my server!');
});

const server = app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

const io = socket(server, {
  cors: {
    origin: "http://localhost:3000",
    credentials: true,
  },
});

global.onlineUsers = new Map();
io.on("connection", (socket) => {

  global.chatSocket = socket;
  socket.on("add-user", (userId) => {
      onlineUsers.set(userId, socket.id);
  });

  socket.on("send-msg", (data) => {
    console.log(data);
    const sendUserSocket = onlineUsers.get(data.to);
    if (sendUserSocket) {
      console.log("ADA");
      socket.to(sendUserSocket).emit("msg-receive", data);
    }
  });

  socket.on("disconnect", () => {
    socket.broadcast.emit("callEnded");
  });

  socket.on("callUser", (data) => {
    console.log(" " + data.from + " " + data.signalData);
    const sendUserSocket = onlineUsers.get(data.userToCall);
    // console.log(data.signalData);
    socket.to(sendUserSocket).emit("callUser", {chatId: data.chatId, lastMessage: data.lastMessage, timestamp: data.timestamp, signal: data.signalData, from: data.from, username: data.username});
  });

  socket.on("answerCall", (data) => {
    const sendUserSocket = onlineUsers.get(data.to);
    socket.to(sendUserSocket).emit("callAccepted", data.signal);
  });
  
});
