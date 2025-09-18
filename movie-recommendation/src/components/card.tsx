import Link from "next/link";

type CardProps = {
  title: string;
  description: string;
  imageUrl?: string;
  linkUrl?: string;
};

export default function Card({
  title,
  description,
  imageUrl = "/image/p1.png",
  linkUrl = "#",
}: CardProps) {
  return (
    <div className="flex flex-col rounded-lg overflow-hidden bg-white shadow-lg hover:shadow-xl transition-shadow duration-300 border border-gray-200">
      {/* Image */}
      <Link href={linkUrl} scroll={false} className="overflow-hidden">
        <img
          src={imageUrl}
          alt={title}
          className="w-full h-72 object-cover hover:scale-105 transition-transform duration-500"
        />
      </Link>

      {/* Content */}
      <div className="p-4 flex flex-col flex-grow">
        <Link href={linkUrl} scroll={false} className="no-underline">
          <h2 className="text-xl font-bold text-gray-900 mb-2">{title}</h2>
        </Link>
        <p className="text-gray-600 mb-4 text-sm flex-grow">{description}</p>

        {/* CTA Button */}
        <Link
          href={linkUrl}
          scroll={false}
          className="self-start px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
        >
          Learn More →
        </Link>
      </div>
    </div>
  );
}
