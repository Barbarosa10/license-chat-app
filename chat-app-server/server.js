const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const socket = require("socket.io");
const _ = require("lodash");

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
    origin: "http://localhost:3001",
    credentials: true,
  },
});


users = new Map();
onlineUsers = new Map();
io.on("connection", (socket) => {
  const userId = socket.handshake.query.userId;

  // console.log(userId);
  // console.log(socket.id);
  // if(userId != undefined)
  //   users.set(userId, socket.id);
  console.log("user-videoooo: ");
  console.log(users);

  socket.on("add-user", (userId) => {
      onlineUsers.set(userId, socket.id);
  });
  socket.on("add-user-video", (userId) => {
    console.log("user-video: ");

    users.set(userId, socket.id);
    console.log(users);
    // console.log(userId);
    // console.log(socket.id);
  });


  socket.on("send-msg", (data) => {
    // console.log(data);
    const sendUserSocket = onlineUsers.get(data.to);
    if (sendUserSocket) {
      // console.log("ADA");
      socket.to(sendUserSocket).emit("msg-receive", data);
    }
  });

  socket.on("open-socket", (data) => {
    console.log(data);
    console.log(onlineUsers);
    const sendUserSocket = onlineUsers.get(data.to);
    console.log(sendUserSocket);
    if (sendUserSocket) {
      // console.log("ADA");
      socket.to(sendUserSocket).emit("open-socket", {chatId: data.chatId, to: data.to, from: data.from});
    }
  });
  socket.on("socket-opened", (data) => {
    console.log(onlineUsers);
    console.log(data);
    const sendUserSocket = onlineUsers.get(data.to);
    console.log("opopened");
    console.log(sendUserSocket);
    if (sendUserSocket) {
      console.log("socket-opened");
      socket.to(sendUserSocket).emit("socket-opened", data);
    }
  });

  socket.on("end-call", (data) => {
    console.log(data);
    console.log(users);
    const sendUserSocket = users.get(data.to);
    if (sendUserSocket) {
      socket.to(sendUserSocket).emit("callEnded", data);
    }
  });

  socket.on("disconnect", () => {
    socket.broadcast.emit("disconnected");
    // _.pull(users, socket.id)
  });

  socket.on("callUser", (data) => {
    console.log(users);
    // console.log(" " + data.from + " " + data.signalData);
    const sendUserSocket = users.get(data.userToCall);
    // console.log(data.signalData);
    socket.to(sendUserSocket).emit("callUser", {chatId: data.chatId, lastMessage: data.lastMessage, timestamp: data.timestamp, signal: data.signalData, from: data.from, username: data.username});
  });

  socket.on("answerCall", (data) => {
    const sendUserSocket = users.get(data.to);
    socket.to(sendUserSocket).emit("callAccepted", data.signal);
  });
  
});
