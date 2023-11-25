const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

const authRoutes = require("./routes/auth_users");

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

app.get('/', (req, res) => {
  res.send('Welcome to my server!');
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});