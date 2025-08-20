FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install --legacy-peer-deps

COPY . .

# Ensure local node binaries are in PATH
ENV PATH /app/node_modules/.bin:$PATH

EXPOSE 3000
CMD ["npx", "next", "dev", "-p", "3000", "--hostname", "0.0.0.0"]

