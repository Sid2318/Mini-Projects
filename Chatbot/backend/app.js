const express = require('express')
const app = express();

app.use((req,res,next) => {
  console.log(`Request received: ${req.method} ${req.url}`);
  next();
});

app.listen(process.env.PORT || 5000, () => console.log(`Server running on port http://localhost:${process.env.PORT || 5000}`));
