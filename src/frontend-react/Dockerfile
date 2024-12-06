# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source files
COPY . .

# Build the Next.js app
RUN npm run build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Copy built assets from build stage
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/package-lock.json ./package-lock.json
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public

# Install only production dependencies
RUN npm ci --only=production

# Expose the listening port
EXPOSE 3000

# Run the Next.js app
CMD ["npm", "start"]